#coding: utf-8
import os

from chainer.training import StandardUpdater
from chainer import Variable

import chainer.functions as F
import numpy as np
from PIL import Image

from loss import dis_loss, gen_loss, gen_semi_loss
from chainer.backends import cuda
from functions import onehot2label


class AdvSemiSeg_Updater(StandardUpdater):
    def __init__(self, opt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt = opt

        self.num_saved_img = 0
        self.learn_from_unlabel = False
        self.img4save = None
        self.semi_img4save = None

    def update_core(self):
        g_opt = self.get_optimizer('gen')
        d_opt = self.get_optimizer('dis')

        #predict
        x, real_g = self.real_batch('main')
        fake_g = g_opt.target(x)

        self.img4save = [cuda.to_cpu(x.array[0]),
                         cuda.to_cpu(real_g.array[0]),
                         cuda.to_cpu(fake_g.array[0])]

        real_d = d_opt.target(real_g)
        fake_d = d_opt.target(fake_g)

        #generator loss
        g_loss = gen_loss(self.opt, fake_d, real_g, fake_g, observer=g_opt.target)
        g_opt.target.cleargrads()
        g_loss.backward()
        g_opt.update()

        #discriminator loss
        x.unchain_backward()
        fake_g.unchain_backward()

        d_loss = dis_loss(self.opt, real_d, fake_d, observer=d_opt.target)
        d_opt.target.cleargrads()
        d_loss.backward()
        d_opt.update()

        if self.learn_from_unlabel:
            #predict
            unlabel_x, _ = self.real_batch('semi')
            unlabel_g = g_opt.target(unlabel_x)
            unlabel_d = d_opt.target(unlabel_g)

            self.semi_img4save = (cuda.to_cpu(unlabel_x.array[0]),
                                  None,
                                  cuda.to_cpu(unlabel_g.array[0]))

            #semi-supervised loss
            semi_loss = gen_semi_loss(self.opt, unlabel_d, unlabel_g, observer=g_opt.target)
            g_opt.target.cleargrads()
            semi_loss.backward()
            g_opt.update()

    def real_batch(self, iter_key='main'):
        batch = self.get_iterator(iter_key).next()
        batch = self.converter(batch, self.device)

        if isinstance(batch, tuple) or isinstance(batch, list):
            x, t = batch

            #16bit -> 32bit (not use tensor core)
            x = Variable(x.astype('float32'))
            t = Variable(t.astype('float32'))

            return x, t

        x = Variable(batch.astype('float32'))

        return x, None

    def save_img(self):
        if self.img4save is None:
            return

        lines = [self.img4save]
        if self.learn_from_unlabel:
            lines.append(self.semi_img4save)

        tile_img = None
        for l in lines:
            for i, sect in enumerate(l):
                if sect is None:
                    sect = np.zeros_like(l[0].shape)

                if i != 0:
                    sect = onehot2label(sect)

                l[i] = np.transpose(sect, (1, 2, 0))

            l = np.concatenate(l, axis=1)
            if tile_img is None:
                tile_img = l
            else:
                tile_img = np.concatenate((tile_img, l), axis=0)

        out = np.uint8(tile_img * 255)
        out = Image.fromarray(out)

        out_dir_name = self.opt.out_dir + '/out_img'
        os.makedirs(out_dir_name, exist_ok=True)
        out.save(out_dir_name + '/' + str(self.num_saved_img) + '.png')
        self.num_saved_img += 1

    def ignition_semi_learning(self):
        self.learn_from_unlabel = True