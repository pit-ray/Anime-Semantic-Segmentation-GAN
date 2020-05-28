#coding: utf-8
import os
import sys

import chainer
import chainer.training.extensions as ex
import cupy as cp
import numpy as np
from chainer import iterators, optimizer_hooks, optimizers, training
from PIL import Image
from glob import glob

import cv2
from datasets import get_dataset, get_unlabel_dataset
from functions import onehot2label
from generator import ResNetDeepLab
from options import get_options

#in default, not removing
def remove_noise(img, ksize=5):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_mask = cv2.medianBlur(img, ksize)

    indexes = np.where((img_mask == [0, 0, 0]).all(axis=2))
    img_mask[indexes] = img[indexes]

    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)

    return img_mask

def img_save(x, path):
    img_array = np.transpose(x, (1, 2, 0))
    img_array = np.uint8(img_array * 255)
    img = Image.fromarray(img_array)
    img.save(path)

def is_exist_color(img, rgb_list, threshold_num=1):
    class_color = np.array(rgb_list).astype('uint8')
    class_color = class_color.reshape(1, 1, 3)
    class_color = np.repeat(class_color, img.shape[0], axis=0)
    class_color = np.repeat(class_color, img.shape[1], axis=1)

    mask = np.sum(img == rgb_list, axis=2) == 3
    out = np.sum(mask) >= threshold_num

    return out

def main():
    out_dir = 'predict_to'
    in_dir = 'predict_from'
    batch_size = 1
    gen_npz = 'pretrained/gen.npz'

    opt = get_options()

    gen = ResNetDeepLab(opt)
    gen.to_gpu(0)
    chainer.serializers.load_npz(gen_npz, gen)
    gen.to_cpu()

    num = 0
    ksize = 5

    os.makedirs(out_dir, exist_ok=True)

    files = glob(in_dir + '/*.*')

    for filename in files:
        print(filename)

        img_array = np.array(Image.open(filename), dtype='float32')
        img_array = img_array.transpose((2, 0, 1)) / 255
        x = chainer.Variable(img_array[np.newaxis, :3, :, :])

        out = gen(x)

        onehot = out.array[0]
        x = x.array[0]

        out = onehot2label(onehot)

        bg_onehot = np.argmax(onehot, axis=0)
        bg_onehot = bg_onehot == 4
        bg_threshold = 0.5

        bg_num = x * bg_onehot
        bg_num = bg_num > bg_threshold
        bg_num = np.sum(bg_num, axis=0)
        bg_num = np.sum(bg_num == 3)

        bg_ratio = bg_num / np.sum(bg_onehot)

        if bg_ratio < 0.6:
            print('bg is black')
            continue

        out = np.transpose(out * 255, (1, 2, 0)).astype('uint8')
        #out = remove_noise(out, ksize=ksize)

        #exist eye ?
        if not is_exist_color(out, [255, 0, 0], threshold_num=32):
            print('not exist eye')
            continue

        #exist face ?
        if not is_exist_color(out, [0, 255, 0], threshold_num=100):
            print('not exist face')
            continue

        #exist hair ?
        if not is_exist_color(out, [0, 0, 255], threshold_num=100):
            print('not exist hair')
            continue

        x = np.transpose(x * 255, (1, 2, 0)).astype('uint8')

        out_img = np.concatenate((x, out), axis=1)
        img = Image.fromarray(out_img)
        path = out_dir + '/' + str(num) + '.png'
        img.save(path)

        num += 1

if __name__ == '__main__':
    main()
