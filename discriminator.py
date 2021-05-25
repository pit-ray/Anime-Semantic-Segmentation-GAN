# coding: utf-8
import chainer.functions as F
import chainer.links as L
from chainer.initializers import HeNormal, Normal
from chainer import Chain


from architecture import PixelShuffler
from spectral_norms import define_conv, define_deconv


class FCN(Chain):
    def __init__(self, opt):
        super().__init__()
    
        he_w = HeNormal()
        xavier_w = Normal()
        ndf = opt.ndf
        with self.init_scope():
            # [input] nclass x 256 x 256
            self.c1 = define_conv(opt)(opt.class_num, ndf, ksize=3, stride=2, pad=1, initialW=he_w)

            # [input] ndf x 128 x 128
            self.c2 = define_conv(opt)(ndf, ndf * 2, ksize=3, stride=2, pad=1, initialW=he_w)
            self.c2_norm = L.BatchNormalization(size=ndf * 2)

            # [input] ndf*2 x 64 x 64
            self.c3 = define_conv(opt)(ndf * 2, ndf * 4, ksize=3, stride=2, pad=1, initialW=he_w)
            self.c3_norm = L.BatchNormalization(size=ndf * 4)

            # [input] ndf*4 x 32 x 32
            self.c4 = define_conv(opt)(ndf * 4, ndf * 8, ksize=3, stride=2, pad=1, initialW=he_w)
            self.c4_norm = L.BatchNormalization(size=ndf * 8)

            # [input] ndf*8 x 16 x 16
            self.c5 = define_conv(opt)(ndf * 8, 1, ksize=3, stride=2, pad=1, initialW=he_w)

            # [input] 1 x 8 x 8
            self.upscale = define_deconv(opt)(1, 1, ksize=32, stride=32, initialW=xavier_w)
            # [output] 1 x 256 x 256

        self.activation = F.leaky_relu

    def __call__(self, x):
        h = self.c1(x)
        h = self.activation(h)

        h = self.c2(h)
        h = self.c2_norm(h)
        h = self.activation(h)

        h = self.c3(h)
        h = self.c3_norm(h)
        h = self.activation(h)
        
        h = self.c4(h)
        h = self.c4_norm(h)
        h = self.activation(h)

        h = self.c5(h)
        h = self.activation(h)

        out = self.upscale(h)
        return out
