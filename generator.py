#coding: utf-8
from chainer import Chain
from chainer.backends import cuda
from chainer.initializers import HeNormal
import chainer.functions as F
import chainer.links as L

from architecture import ASPP, PixelShuffler
from spectral_norms import define_conv, define_deconv
from atrous_conv import define_atrous_conv

class ResNetDeepLab(Chain):
    def __init__(self, opt):
        super().__init__()

        he_w = HeNormal()
        with self.init_scope():
            # This ResNet101 use a pre-trained caffemodel that can be downloaded at GitHub
            # <https://github.com/KaimingHe/deep-residual-networks>.
            self.resnet101 = L.ResNet101Layers()
            self.use_layer = ('res3', 512)
            nf = self.use_layer[1]

            self.c1 = define_atrous_conv(opt)(nf, nf, ksize=3, rate=2, initialW=he_w)
            self.norm1 = L.BatchNormalization(nf)

            self.c2 = define_atrous_conv(opt)(nf, nf, ksize=3, rate=4, initialW=he_w)
            self.norm2 = L.BatchNormalization(nf)

            self.aspp = ASPP(opt, nf, input_resolution=32)
            self.up1 = PixelShuffler(opt, nf, nf // 2, rate=2) #32 -> 64
            self.up2 = PixelShuffler(opt, nf // 2, nf // 4, rate=2) #64 -> 128
            self.up3 = PixelShuffler(opt, nf // 4, nf // 8, rate=2) # 128 -> 256
            self.to_class = define_conv(opt)(nf // 8, opt.class_num, ksize=3, pad=1, initialW=he_w)

        self.activation = F.leaky_relu

    def prepare(self, variable_img):
        #out = F.resize_images(variable_img, (224, 224))
        out = variable_img
        #out = (out + 1) * 0.5
        out = out[:, ::-1, :, :]
        out = F.transpose(out, (0, 2, 3, 1))

        out *= 255
        xp = cuda.get_array_module(variable_img.array)
        out -= xp.array([103.063, 115.903, 123.152], dtype=variable_img.dtype)

        out = F.transpose(out, (0, 3, 1, 2))

        return out

    def __call__(self, x):
        x = self.prepare(x)
        h = self.resnet101(x, [self.use_layer[0]])[self.use_layer[0]]
        h = self.activation(h)

        h = self.c1(h)
        h = self.norm1(h)
        h = self.activation(h)

        h = self.c2(h)
        h = self.norm2(h)
        h = self.activation(h)

        h = self.aspp(h)

        h = self.up1(h)
        h = self.activation(h)

        h = self.up2(h)
        h = self.activation(h)

        h = self.up3(h)
        h = self.activation(h)

        out = self.to_class(h)
        out = F.softmax(out, axis=1)

        return out


class DilatedFCN(Chain):
    def __init__(self, opt):
        super().__init__()

        he_w = HeNormal()
        down_sampling_num = 3

        ngf = opt.ngf

        with self.init_scope():
            #[input] 3 x 256 x 256
            self.c1 = define_conv(opt)(opt.img_shape[0], ngf, ksize=4, stride=2, pad=1, initialW=he_w)
            self.norm1 = L.BatchNormalization(ngf)

            #[input] ngf x 128 x 128
            self.c2 = define_conv(opt)(ngf, ngf * 2, ksize=4, stride=2, pad=1, initialW=he_w)
            self.norm2 = L.BatchNormalization(ngf * 2)

            #[input] ngf*2 x 64 x 64
            self.c3 = define_conv(opt)(ngf * 2, ngf * 4, ksize=4, stride=2, pad=1, initialW=he_w)
            self.norm3 = L.BatchNormalization(ngf * 4)

            #[input] ngf*4 x 32 x 32
            self.a1 = define_atrous_conv(opt)(ngf * 4, ngf * 4, ksize=3, rate=2, initialW=he_w)
            self.norm4 = L.BatchNormalization(ngf * 4)

            #[input] ngf*4 x 32 x 32
            self.a2 = define_atrous_conv(opt)(ngf * 4, ngf * 4, ksize=3, rate=4, initialW=he_w)
            self.norm5 = L.BatchNormalization(ngf * 4)

            #[input] ngf*4 x 32 x 32
            resolution = max(opt.img_shape[1], opt.img_shape[2]) // 2 ** down_sampling_num
            self.aspp = ASPP(opt, ngf * 4, input_resolution=resolution)

            #[input] ngf*4 x 32 x 32
            self.up1 = PixelShuffler(opt, ngf * 4, ngf * 2, rate=2) #64 -> 128
            self.up2 = PixelShuffler(opt, ngf * 2, ngf, rate=2) # 128 -> 256
            self.to_class = define_conv(opt)(ngf, opt.class_num, ksize=3, pad=1, initialW=he_w)
            #[output] class_num x 256 x 256

        self.activation = F.relu

    def __call__(self, x):
        h = self.c1(x)
        h = self.norm1(h)
        h = self.activation(h)

        h = self.c2(h)
        h = self.norm2(h)
        h = self.activation(h)

        h = self.c3(h)
        h = self.norm3(h)
        h = self.activation(h)

        h = self.a1(h)
        h = self.norm4(h)
        h = self.activation(h)

        h = self.a2(h)
        h = self.norm5(h)
        h = self.activation(h)

        h = self.aspp(h)

        h = self.up1(h)
        h = self.activation(h)

        h = self.up2(h)
        h = self.activation(h)

        out = self.to_class(h)
        out = F.softmax(out, axis=1)

        return out


class UNet(Chain):
    def __init__(self, opt):
        super().__init__()

        he_w = HeNormal()
        ngf = opt.ngf
        with self.init_scope():
            #Encoder
            #[input] 3 x 256 x 256 
            self.e1 = define_conv(opt)(opt.input_ch, ngf, ksize=3, stride=1, pad=1, initialW=he_w)
            self.e1_bn = L.BatchNormalization(ngf)

            #[input] ngf x 256 x 256 
            self.e2 = define_conv(opt)(ngf, ngf * 2, ksize=4, stride=2, pad=1, initialW=he_w)
            self.e2_bn = L.BatchNormalization(ngf * 2)

            #[input] ngf*2 x 128 x 128 
            self.e3 = define_conv(opt)(ngf * 2, ngf * 4, ksize=4, stride=2, pad=1, initialW=he_w)
            self.e3_bn = L.BatchNormalization(ngf * 4)

            #[input] ngf*4 x 64 x 64 
            self.e4 = define_conv(opt)(ngf * 4, ngf * 8, ksize=4, stride=2, pad=1, initialW=he_w)
            self.e4_bn = L.BatchNormalization(ngf * 8)

            #[input] ngf*8 256 x 32 x 32 
            self.e5 = define_conv(opt)(ngf * 8, ngf * 16, ksize=4, stride=2, pad=1, initialW=he_w)
            self.e5_bn = L.BatchNormalization(ngf * 16)

            #Decoder
            #[input] ngf*16 x 16 x 16 
            self.d1 = L.Deconvolution2D(ngf * 16, ngf * 8, ksize=4, stride=2, pad=1, initialW=he_w)
            self.d1_bn = L.BatchNormalization(ngf * 8)

            #[input] ngf*8*2 x 32 x 32 (concat)
            self.d2 = L.Deconvolution2D(ngf * 8 * 2, ngf * 4, ksize=4, stride=2, pad=1, initialW=he_w)
            self.d2_bn = L.BatchNormalization(ngf * 4)

            #[input] ngf*4*2 x 64 x 64 (concat)
            self.d3 = L.Deconvolution2D(ngf * 4 * 2, ngf * 2, ksize=4, stride=2, pad=1, initialW=he_w)
            self.d3_bn = L.BatchNormalization(ngf * 2)

            #[input] ngf*2*2 x 128 x 128 (concat)
            self.d4 = L.Deconvolution2D(ngf * 2 * 2, ngf, ksize=4, stride=2, pad=1, initialW=he_w)
            self.d4_bn = L.BatchNormalization(ngf)

            #[input] ngf x 256 x 256
            self.to_class = define_conv(opt)(ngf, opt.nclass, ksize=3, pad=1, initialW=he_w)
            #[output] nclass x 256 x 256

        self.activation = F.relu

    def __call__(self, x):
        #Encoder
        eh1 = self.e1(x)
        eh1 = self.e1_bn(eh1)
        eh1 = self.activation(eh1)

        eh2 = self.e2(eh1)
        eh2 = self.e2_bn(eh2)
        eh2 = self.activation(eh2)

        eh3 = self.e3(eh2)
        eh3 = self.e3_bn(eh3)
        eh3 = self.activation(eh3)

        eh4 = self.e4(eh3)
        eh4 = self.e4_bn(eh4)
        eh4 = self.activation(eh4)

        eh5 = self.e5(eh4)
        eh5 = self.e5_bn(eh5)
        eh5 = self.activation(eh5)

        #Decoder
        dh1 = self.d1(eh5)
        dh1 = self.d1_bn(dh1)
        dh1 = F.dropout(dh1)
        dh1 = self.activation(dh1)

        dh2 = F.concat((eh4, dh1), axis=1)
        dh2 = self.d2(dh2)
        dh2 = self.d2_bn(dh2)
        dh2 = F.dropout(dh2)
        dh2 = self.activation(dh2)

        dh3 = F.concat((eh3, dh2), axis=1)
        dh3 = self.d3(dh3)
        dh3 = self.d3_bn(dh3)
        dh3 = F.dropout(dh3)
        dh3 = self.activation(dh3)

        dh4 = F.concat((eh2, dh3), axis=1)
        dh4 = self.d4(dh4)
        dh4 = self.d4_bn(dh4)
        dh4 = self.activation(dh4)

        out = self.to_class(dh4)
        out = F.softmax(out, axis=1)

        return out