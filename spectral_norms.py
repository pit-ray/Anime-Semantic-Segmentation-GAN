#coding: utf-8
import chainer.functions as F
import chainer.links as L
from chainer.link_hooks import SpectralNormalization
from chainer import Chain

#wraping chainer.links.Convolution2D / chainer.links.Deconvolution2D
#this is based on
#Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.
#Spectral Normalization for Generative Adversarial Networks.
#arXiv preprint arXiv:1802.05957, 2018.

#However, my implementation is for the archtecture which chainer.link_hooks is unavaiable, for example WebDNN or old Chainer.
#Thus, if you dont have specially reason, I recommend you to use chainer.link_hooks.SpectralNormalization().

#learnable spectral norm
def spectral_normalize(weight, init_u):
    W = weight.reshape(weight.shape[0], -1) #C x N
    v = F.normalize(F.matmul(W, init_u, transa=True), eps=1e-12, axis=0) #N x C * C x 1 -> N x 1
    u = F.normalize(F.matmul(W, v), eps=1e-12, axis=0) #C x N * N x 1 -> C x 1
    sigma = F.matmul(F.matmul(u, W, transa=True), v) #1 x C * C x N * N x -> 1 x 1 (spectral norm)
    return weight / sigma


class SNConv(L.Convolution2D):
    def __init__(self, input_ch, output_ch, ksize=None, stride=1, pad=0,
        nobias=False, initialW=None, initial_bias=None, dilate=1):
        super().__init__(self, input_ch, output_ch, ksize=ksize, stride=stride, pad=pad,
            nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate)

        self.init_u = self.xp.random.normal(size=(output_ch, )).astype(self.W.dtype)
        self.register_persistent('init_u')

    def __call__(self, x):
        return F.convolution_2d(x, spectral_normalize(self.W, self.init_u),
            b=self.b, stride=self.stride, pad=self.pad)


class SNDeconv(L.Deconvolution2D):
    def __init__(self, input_ch, output_ch, ksize=None, stride=1, pad=0,
        nobias=False, initialW=None, initial_bias=None, outsize=None):
        super().__init__(self, input_ch, output_ch, ksize=ksize, stride=stride, pad=pad,
            nobias=nobias, initialW=initialW, initial_bias=initial_bias, outsize=outsize)

        self.init_u = self.xp.random.normal(size=(input_ch, )).astype(self.W.dtype)
        self.register_persistent('init_u')

    def __call__(self, x):
        return F.deconvolution_2d(x, spectral_normalize(self.W, self.init_u),
            b=self.b, stride=self.stride, pad=self.pad, outsize=self.outsize)


#This Spectral Normalization is optimized to Chainer.
class SNHookConv(Chain):
    def __init__(self, input_ch, out_ch, ksize=None, stride=1, pad=0,
        nobias=False, initialW=None, initial_bias=None, dilate=1):
        super().__init__()

        with self.init_scope():
            self.c =L.Convolution2D(input_ch, out_ch, ksize=ksize, stride=stride, pad=pad,
                nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate)
            self.c.add_hook(SpectralNormalization())

    def __call__(self, x):
        return self.c(x)

class SNHookDeconv(Chain):
    def __init__(self, input_ch, out_ch, ksize=None, stride=1, pad=0,
        nobias=False, initialW=None, initial_bias=None, outsize=None):
        super().__init__()

        with self.init_scope():
            self.c =L.Deconvolution2D(input_ch, out_ch, ksize=ksize, stride=stride, pad=pad,
                nobias=nobias, initialW=initialW, initial_bias=initial_bias, outsize=outsize)
            self.c.add_hook(SpectralNormalization())

    def __call__(self, x):
        return self.c(x)


def define_conv(opt):
    if opt.conv_norm == 'original':
        return L.Convolution2D
    
    if opt.conv_norm == 'spectral_norm':
        return SNConv
    
    if opt.conv_norm == 'spectral_norm_hook':
        return SNHookConv


def define_deconv(opt):
    if opt.conv_norm == 'original':
        return L.Deconvolution2D
    
    if opt.conv_norm == 'spectral_norm':
        return SNDeconv
    
    if opt.conv_norm == 'spectral_norm_hook':
        return SNHookDeconv