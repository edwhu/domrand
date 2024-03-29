import math
import os
import time
from collections import defaultdict

import mujoco_py
import numpy as np
import quaternion
import skimage
import yaml
from mujoco_py import MjSim, MjViewer, functions, load_model_from_path
from mujoco_py.modder import (BaseModder, CameraModder, LightModder,
                              MaterialModder)

from domrand.define_flags import FLAGS
from domrand.utils.data import get_real_cam_pos
from domrand.utils.image import display_image, preproc_image
from domrand.utils.modder import TextureModder
from domrand.utils.sim import (Range, Range3D,  # object type things
                               jitter_angle, jitter_quat, look_at, random_quat,
                               rto3d, sample, sample_geom_type, sample_joints,
                               sample_light_dir, sample_quat, sample_xyz)


# GLOSSARY:
# gid = geom_id
# bid = body_id
class SimManager(object):
    """Object to generate sequence of images with their transforms"""
    def __init__(self, filepath, random_params={}, gpu_render=False, gui=False, display_data=False):
        self.model = load_model_from_path(filepath)
        self.sim = MjSim(self.model)
        self.filepath = filepath
        self.gui = gui
        self.display_data = display_data
        # Take the default random params and update anything we need
        self.RANDOM_PARAMS = {}
        self.RANDOM_PARAMS.update(random_params)

        if gpu_render:
            self.viewer = MjViewer(self.sim)
        else:
            self.viewer = None

        # Get start state of params to slightly jitter later
        self.START_GEOM_POS = self.model.geom_pos.copy()
        self.START_GEOM_SIZE = self.model.geom_size.copy()
        self.START_GEOM_QUAT = self.model.geom_quat.copy()
        self.START_BODY_POS = self.model.body_pos.copy()
        self.START_BODY_QUAT = self.model.body_quat.copy()
        self.START_MATID = self.model.geom_matid.copy()
        #self.FLOOR_OFFSET = self.model.body_pos[self.model.body_name2id('floor')]

        self.tex_modder = TextureModder(self.sim)
        self.tex_modder.whiten_materials()  # ensures materials won't impact colors
        self.cam_modder = CameraModder(self.sim)
        self.light_modder = LightModder(self.sim)
        self.start_obj_pose = self.sim.data.get_joint_qpos('object:joint').copy()


    def get_data(self, num_images=10):
        """
        Returns camera intrinsics, and a sequence of images, pose transforms, and
        camera transforms
        """
        # randomize the scene
        self._rand_textures()
        self._rand_lights()
        self._rand_object()
        self._rand_walls()
        self._rand_distract()
        sequence = defaultdict(list)
        context = {}
        # object pose
        obj_pose, robot_pose = self._get_ground_truth()
        context["obj_world_pose"] = obj_pose
        context["robot_world_pose"] = robot_pose
        self._cam_step = 0
        self._cam_choices = np.array([[-1.75, 0, 2], [-1.75, 0, 1.62], [-1.3, 1.7, 1.62]])
        self._curr_cam_pos = self._cam_choices[0]
        for i in range(num_images):
             self._next_camera()
             self._forward()
             img = self._get_cam_frame()
             sequence["img"].append(img)
             cam_pos = self.cam_modder.get_pos("camera1")
             cam_quat = self.cam_modder.get_quat("camera1")
             cam_pose = np.concatenate([cam_pos, cam_quat]).astype(np.float32)
             sequence["cam_pose"].append(cam_pose)

        cam_id = self.cam_modder.get_camid("camera1")
        fovy = self.sim.model.cam_fovy[cam_id]
        width, height = 640, 480
        f = 0.5 * height / math.tan(fovy * math.pi / 360)
        camera_intrinsics =  np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        context['cam_matrix'] = camera_intrinsics
        return context, sequence

    def _forward(self):
        """Advances simulator a step (NECESSARY TO MAKE CAMERA AND LIGHT MODDING WORK)
        And add some visualization"""
        self.sim.forward()
        if self.viewer and self.gui:
            # Get angle of camera and display it
            quat = np.quaternion(*self.model.cam_quat[0])
            ypr = quaternion.as_euler_angles(quat) * 180 / np.pi
            cam_pos = self.model.cam_pos[0]
            cam_fovy = self.model.cam_fovy[0]
            #self.viewer.add_marker(pos=cam_pos, label="CAM: {}{}".format(cam_pos, ypr))
            #self.viewer.add_marker(pos=cam_pos, label="CAM: {}".format(ypr))
            #self.viewer.add_marker(pos=cam_pos, label="CAM: {}".format(cam_pos))
            self.viewer.add_marker(pos=cam_pos, label="FOVY: {}, CAM: {}".format(cam_fovy, cam_pos))
            self.viewer.render()

    def _get_ground_truth(self):
        """
        Return the  position to the robot, and quaternion to the robot quaternion
        7 dim total
        """
        robot_gid = self.sim.model.geom_name2id('base_link')
        obj_gid = self.sim.model.geom_name2id('object')
        # only x and y pos needed
        obj_world_pos = self.sim.data.geom_xpos[obj_gid]
        robot_world_pos = self.sim.data.geom_xpos[robot_gid]
        # obj_pos_in_robot_frame = (self.sim.data.geom_xpos[obj_gid] - self.sim.data.geom_xpos[robot_gid])[:2]

        # robot_quat = quaternion.as_quat_array(self.model.geom_quat[robot_gid].copy())
        obj_world_quat = self.model.geom_quat[obj_gid].copy()
        robot_world_quat = self.model.geom_quat[robot_gid].copy()
        # # want quat of obj relative to robot frame
        # # obj_q = robot_q * localrot
        # # robot_q.inv * obj_q = localrot
        # rel_quat = quaternion.as_float_array(robot_quat.inverse() * obj_quat)
        # pose = np.concatenate([obj_pos_in_robot_frame, rel_quat]).astype(np.float32)
        obj_pose = self.sim.data.get_joint_qpos('object:joint').copy()
        robot_pose = np.concatenate([robot_world_pos, robot_world_quat]).astype(np.float32)
        return obj_pose, robot_pose

    def _get_cam_frame(self, ground_truth=None):
        """Grab an image from the camera (224, 244, 3) to feed into CNN"""
        #IMAGE_NOISE_RVARIANCE = Range(0.0, 0.0001)
        cam_img = self.sim.render(640, 480, camera_name='camera1')[::-1, :, :] # Rendered images are upside-down.
        # make camera crop be more like kinect
        #cam_img = self.sim.render(854, 480, camera_name='camera1')[::-1, 107:-107, :] # Rendered images are upside-down.

        #image_noise_variance = sample(IMAGE_NOISE_RVARIANCE)
        #cam_img = (skimage.util.random_noise(cam_img, mode='gaussian', var=image_noise_variance) * 255).astype(np.uint8)

        if self.display_data:
            print(ground_truth)
            #label = str(ground_truth[3:6])
            display_image(cam_img, mode='preproc')#, label)

        # cam_img = preproc_image(cam_img)
        return cam_img

    def _randomize(self):
        self._rand_textures()
        self._rand_camera()
        self._rand_lights()
        #self._rand_robot()
        self._rand_object()
        self._rand_walls()
        self._rand_distract()

    def _rand_textures(self):
        """Randomize all the textures in the scene, including the skybox"""
        bright = np.random.binomial(1, 0.5)
        for name in self.sim.model.geom_names + ('skybox',):
            self.tex_modder.rand_all(name)
            if bright:
                self.tex_modder.brighten(name, np.random.randint(0,150))

    def _rand_camera(self):
        """Randomize pos, orientation, and fov of camera
        real camera pos is -1.75, 0, 1.62
        FOVY:
        Kinect2 is 53.8
        ASUS is 45
        https://www.asus.com/us/3D-Sensor/Xtion_PRO_LIVE/specifications/
        http://smeenk.com/kinect-field-of-view-comparison/
        """
        # Params
        FOVY_R = Range(40, 50)
        #X = Range(-3, -1)
        #Y = Range(-1, 3)
        #Z = Range(1, 2)
        #C_R3D = Range3D(X, Y, Z)
        #cam_pos = sample_xyz(C_R3D)
        #L_R3D = rto3d([-0.1, 0.1])

        C_R3D = Range3D([-0.07,0.07], [-0.07,0.07], [-0.07,0.07])
        ANG3 = Range3D([-3,3], [-3,3], [-3,3])

        # Look approximately at the robot, but then randomize the orientation around that
        cam_choices = np.array([[-1.75, 0, 1.62], [-1.3, 1.7, 1.62], [-1.75, 0, 2]])
        cam_pos = cam_choices[np.random.choice(len(cam_choices))]
        # cam_pos = get_real_cam_pos(FLAGS.real_data_path)
        target_id = self.model.body_name2id(FLAGS.look_at)

        cam_off = 0 #sample_xyz(L_R3D)
        target_off = 0 #sample_xyz(L_R3D)
        quat = look_at(cam_pos+cam_off, self.sim.data.body_xpos[target_id]+target_off)
        quat = jitter_angle(quat, ANG3)
        #quat = jitter_quat(quat, 0.01)

        cam_pos += sample_xyz(C_R3D)

        self.cam_modder.set_quat('camera1', quat)
        self.cam_modder.set_pos('camera1', cam_pos)
        self.cam_modder.set_fovy('camera1', 60) # hard code to wide fovy

    def _next_camera(self):
        """Randomize pos, orientation, and fov of camera
        real camera pos is -1.75, 0, 1.62
        FOVY:
        Kinect2 is 53.8
        ASUS is 45
        https://www.asus.com/us/3D-Sensor/Xtion_PRO_LIVE/specifications/
        http://smeenk.com/kinect-field-of-view-comparison/
        """
        # Params
        FOVY_R = Range(40, 50)
        #X = Range(-3, -1)
        #Y = Range(-1, 3)
        #Z = Range(1, 2)
        #C_R3D = Range3D(X, Y, Z)
        #cam_pos = sample_xyz(C_R3D)
        #L_R3D = rto3d([-0.1, 0.1])

        C_R3D = Range3D([-0.07,0.07], [-0.07,0.07], [-0.07,0.07])
        ANG3 = Range3D([-3,3], [-3,3], [-3,3])

        # Look approximately at the robot, but then randomize the orientation around that

        # linearly interpolate to the next camera every K steps
        K = 5
        goal_cam_pos = self._cam_choices[(self._cam_step // K) + 1]
        offset = goal_cam_pos - self._curr_cam_pos
        offset *= (self._cam_step % K) / K
        self._curr_cam_pos += offset
        cam_pos = self._curr_cam_pos
        self._cam_step += 1

        # cam_pos = cam_choices[np.random.choice(len(cam_choices))]
        # cam_pos = get_real_cam_pos(FLAGS.real_data_path)
        target_id = self.model.body_name2id(FLAGS.look_at)

        cam_off = 0 #sample_xyz(L_R3D)
        target_off = 0 #sample_xyz(L_R3D)
        quat = look_at(cam_pos+cam_off, self.sim.data.body_xpos[target_id]+target_off)
        quat = jitter_angle(quat, ANG3)
        #quat = jitter_quat(quat, 0.01)

        cam_pos += sample_xyz(C_R3D)

        self.cam_modder.set_quat('camera1', quat)
        self.cam_modder.set_pos('camera1', cam_pos)
        self.cam_modder.set_fovy('camera1', 60) # hard code to wide fovy

    def _rand_lights(self):
        """Randomize pos, direction, and lights"""
        # light stuff
        #X = Range(-1.5, 1.5)
        #Y = Range(-1.2, 1.2)
        #Z = Range(0, 2.8)
        X = Range(-1.5, -0.5)
        Y = Range(-0.6, 0.6)
        Z = Range(1.0, 1.5)
        LIGHT_R3D = Range3D(X, Y, Z)
        LIGHT_UNIF = Range3D(Range(0,1), Range(0,1), Range(0,1))

        # TODO: also try not altering the light dirs and just keeping them at like -1, or [0, -0.15, -1.0]
        for i, name in enumerate(self.model.light_names):
            lid = self.model.light_name2id(name)
            # random sample 80% of any given light being on
            if lid != 0:
                self.light_modder.set_active(name, sample([0,1]) < 0.8)
                self.light_modder.set_dir(name, sample_light_dir())

            self.light_modder.set_pos(name, sample_xyz(LIGHT_R3D))

            #self.light_modder.set_dir(name, sample_xyz(rto3d([-1,1])))

            #self.light_modder.set_specular(name, sample_xyz(LIGHT_UNIF))
            #self.light_modder.set_diffuse(name, sample_xyz(LIGHT_UNIF))
            #self.light_modder.set_ambient(name, sample_xyz(LIGHT_UNIF))

            spec =    np.array([sample(Range(0.5,1))]*3)
            diffuse = np.array([sample(Range(0.5,1))]*3)
            ambient = np.array([sample(Range(0.5,1))]*3)

            self.light_modder.set_specular(name, spec)
            self.light_modder.set_diffuse(name,  diffuse)
            self.light_modder.set_ambient(name,  ambient)
            #self.model.light_directional[lid] = sample([0,1]) < 0.2
            self.model.light_castshadow[lid] = sample([0,1]) < 0.5

    def _rand_robot(self):
        """Randomize joint angles and jitter orientation"""
        jnt_shape = self.sim.data.qpos.shape
        self.sim.data.qpos[:] = sample_joints(self.model.jnt_range, jnt_shape)

        robot_gid = self.model.geom_name2id('robot_table_link')
        self.model.geom_quat[robot_gid] = jitter_quat(self.START_GEOM_QUAT[robot_gid], 0.01)

    def _rand_object(self):
        obj_gid = self.sim.model.geom_name2id('object')
        obj_bid = self.sim.model.geom_name2id('object')
        table_gid = self.model.geom_name2id('object_table')
        table_bid = self.model.body_name2id('object_table')

        obj_pose = self.start_obj_pose.copy()
        xval = self.model.geom_size[table_gid][0] #- self.model.geom_size[obj_gid][0]
        yval = self.model.geom_size[table_gid][1] #- self.model.geom_size[obj_gid][1]

        O_X = Range(-xval, xval)
        O_Y = Range(-yval, yval)
        O_Z = Range(0, 0)
        O_R3D = Range3D(O_X, O_Y, O_Z)

        newpos = obj_pose[:3] + sample_xyz(O_R3D)
        newquat = jitter_quat(obj_pose[3:], 0.1)
        obj_pose[:3] = newpos
        obj_pose[3:] = newquat
        self.sim.data.set_joint_qpos('object:joint', obj_pose)

        #T_X = Range(-0.1, 0.1)
        #T_Y = Range(-0.1, 0.1)
        #T_Z = Range(-0.1, 0.1)
        #T_R3D = Range3D(T_X, T_Y, T_Z)
        #self.model.body_pos[table_bid] = self.START_BODY_POS[table_bid] + sample_xyz(T_R3D)
        ## randomize orientation a wee bit
        #self.model.geom_quat[table_gid] = jitter_quat(self.START_GEOM_QUAT[table_gid], 0.01)

    def _rand_walls(self):
        wall_bids = {name: self.model.body_name2id(name) for name in ['wall_'+dir for dir in 'nesw']}
        window_gid = self.model.geom_name2id('west_window')
        #floor_gid = self.model.geom_name2id('floor')

        WA_X = Range(-0.2, 0.2)
        WA_Y = Range(-0.2, 0.2)
        WA_Z = Range(-0.1, 0.1)
        WA_R3D = Range3D(WA_X, WA_Y, WA_Z)

        WI_X = Range(-0.1, 0.1)
        WI_Y = Range(0, 0)
        WI_Z = Range(-0.5, 0.5)
        WI_R3D = Range3D(WI_X, WI_Y, WI_Z)

        R = Range(0,0)
        P = Range(-10,10)
        Y = Range(0,0)
        RPY_R = Range3D(R,P,Y)

        #self.model.geom_quat[floor_gid] = jitter_quat(self.START_GEOM_QUAT[floor_gid], 0.01)
        #self.model.geom_pos[floor_gid] = self.START_GEOM_POS[floor_gid] + [0,0,sample(-0.1,0.1)

        self.model.geom_quat[window_gid] = sample_quat(RPY_R)
        #self.model.geom_quat[window_gid] = jitter_quat(self.START_GEOM_QUAT[window_gid], 0.01)
        self.model.geom_pos[window_gid] = self.START_GEOM_POS[window_gid] + sample_xyz(WI_R3D)

        for name in wall_bids:
            gid = wall_bids[name]
            self.model.body_quat[gid] = jitter_quat(self.START_BODY_QUAT[gid], 0.01)
            self.model.body_pos[gid] = self.START_BODY_POS[gid] + sample_xyz(WA_R3D)


    def _rand_distract(self):
        PREFIX = 'distract'
        geom_names = [name for name in self.model.geom_names if name.startswith(PREFIX)]

        # Size range
        SX = Range(0.01, 0.5)
        SY = Range(0.01, 0.9)
        SZ = Range(0.01, 0.5)
        S3D = Range3D(SX, SY, SZ)
        # Back range
        B_PX = Range(-0.5, 2)
        B_PY = Range(-1.5, 2)
        B_PZ = Range(0, 3)
        B_P3D = Range3D(B_PX, B_PY, B_PZ)
        # Front range
        F_PX = Range(-2, -0.5)
        F_PY = Range(-2, 1)
        F_PZ = Range(0, 0.5)
        F_P3D = Range3D(F_PX, F_PY, F_PZ)

        for name in geom_names:
            gid = self.model.geom_name2id(name)
            range = B_P3D if np.random.binomial(1, 0.5) else F_P3D

            self.model.geom_pos[gid] = sample_xyz(range)
            self.model.geom_quat[gid] = random_quat()
            self.model.geom_size[gid] = sample_xyz(S3D, mode='logspace')
            self.model.geom_type[gid] = sample_geom_type()
            self.model.geom_rgba[gid][-1] = np.random.binomial(1, 0.5)


    def _set_visible(self, prefix, range_top, visible):
        """Helper function to set visibility of several objects"""
        if not visible:
            if range_top == 0:
                name = prefix
                gid = self.model.geom_name2id(name)
                self.model.geom_rgba[gid][-1] = 0.0

            for i in range(range_top):
                name = "{}{}".format(prefix, i)
                gid = self.model.geom_name2id(name)
                self.model.geom_rgba[gid][-1] = 0.0
        else:
            if range_top == 0:
                name = prefix
                gid = self.model.geom_name2id(name)
                self.model.geom_rgba[gid][-1] = 1.0

            for i in range(range_top):
                name = "{}{}".format(prefix, i)
                gid = self.model.geom_name2id(name)
                self.model.geom_rgba[gid][-1] = 1.0
