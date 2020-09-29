#!/usr/bin/env python3
import os
import random
import tensorflow as tf
from domrand.define_flags import FLAGS
from domrand.trainer import train_simple, model_eval
from domrand.utils.general import notify

def main():
    print(FLAGS.checkpoint)
    results = train_simple()

    logpath = os.path.join(FLAGS.logpath, '{:0.5f}.txt'.format(results['train_euc']))

    with open(logpath, 'w+') as f:
        f.write(FLAGS.checkpoint)

    if FLAGS.notify:
        notify('Finished run. train: {}'.format(results['train_euc']))

def eval():
    model_eval()

if __name__ == '__main__':
    model_eval()

