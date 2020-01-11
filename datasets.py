#coding: utf-8
import os
from glob import glob

import joblib
import numpy as np
from chainer.datasets.tuple_dataset import TupleDataset
from PIL import Image

from functions import label2onehot


def gamma_correction(img, gamma=2.5):
    return img ** (1 / gamma)


def get_dataset(opt):
    files = glob(opt.dataset_dir + '/*.png')

    os.makedirs('dump', exist_ok=True)
    dump_file = 'dump/datasets_with_label.joblib'

    if os.path.exists(dump_file):
        with open(dump_file, 'rb') as f:
            x, t = joblib.load(f)

        return TupleDataset(x, t)

    x, t = [], []
    for filename in files:
        if not os.path.exists(filename):
            continue

        img_array = np.array(Image.open(filename), dtype='float16')
        img_array = img_array.transpose((2, 0, 1)) / 255

        x_array = img_array[:3, :, :256]
        t_array = img_array[:3, :, 256:]

        #convert to onehot
        t_array = label2onehot(t_array, threshold=0.4, dtype='float16')

        x.append(x_array)
        t.append(t_array)

        #Data-Augmentation
        if opt.augment_data:
            #mirroring
            x.append(x_array[:, :, ::-1])
            t.append(t_array[:, :, ::-1])

            #gamma-correction
            x.append(gamma_correction(x_array, gamma=2.5))
            t.append(t_array)

            #mirroring and gamma correction
            x.append(gamma_correction(x_array[:, :, ::-1], gamma=2.5))
            t.append(t_array[:, :, ::-1])

    with open(dump_file, 'wb') as f:
        joblib.dump((x, t), f, compress=3)

    return TupleDataset(x, t)


def get_unlabel_dataset(opt):
    files = glob(opt.unlabel_dataset_dir + '/*.png')

    os.makedirs('dump', exist_ok=True)
    dump_file = 'dump/datasets_without_label.joblib'

    if os.path.exists(dump_file):
        with open(dump_file, 'rb') as f:
            x = joblib.load(f)
        return x

    x = []
    for filename in files:
        if not os.path.exists(filename):
            continue

        x_array = np.array(Image.open(filename), dtype='float16')
        x_array = x_array.transpose((2, 0, 1)) / 255
        x_array = x_array[:3]
        x.append(x_array)

    with open(dump_file, 'wb') as f:
        joblib.dump(x, f, compress=3)

    return x