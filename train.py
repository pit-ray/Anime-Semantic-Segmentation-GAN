#coding: utf-8
import chainer.training.extensions as ex
from chainer.iterators import SerialIterator
from chainer.optimizer_hooks import WeightDecay
from chainer.optimizers import Adam
from chainer.training import PRIORITY_READER, Trainer

from datasets import get_dataset, get_unlabel_dataset
from discriminator import FCN
from functions import adam_lr_poly
from generator import DilatedFCN, ResNetDeepLab, UNet
from options import get_options
from updater import AdvSemiSeg_Updater


def train(opt):
    if opt.use_cpu:
        device = -1
    else:
        device = 0

    annotated = get_dataset(opt)
    unlabeled = get_unlabel_dataset(opt)

    train_iter = SerialIterator(annotated, opt.batch_size, shuffle=True)
    semi_iter = SerialIterator(unlabeled, opt.batch_size, shuffle=True)

    gen = ResNetDeepLab(opt)
    #gen = DilatedFCN(opt)
    #gen = UNet(opt)

    if device != -1:
        gen.to_gpu(device) #use GPU
    g_optim = Adam(alpha=opt.g_lr, beta1=opt.g_beta1, beta2=opt.g_beta2)
    g_optim.setup(gen)
    if opt.g_weight_decay > 0:
        g_optim.add_hook(WeightDecay(opt.g_weight_decay))

    dis = FCN(opt)
    if device != -1:
        dis.to_gpu(device) #use GPU
    d_optim = Adam(alpha=opt.d_lr, beta1=opt.d_beta1, beta2=opt.d_beta2)
    d_optim.setup(dis)

    updater = AdvSemiSeg_Updater(opt,
        iterator={'main': train_iter, 'semi': semi_iter},
        optimizer={'gen': g_optim, 'dis': d_optim},
        device=device)

    trainer = Trainer(updater, (opt.max_epoch, 'epoch'), out=opt.out_dir)

    #chainer training extensions
    trainer.extend(ex.LogReport(log_name=None, trigger=(1, 'iteration')))
    trainer.extend(ex.ProgressBar((opt.max_epoch, 'epoch'), update_interval=1))

    trainer.extend(ex.PlotReport(['gen/adv_loss', 'dis/adv_loss', 'gen/semi_adv_loss'],
        x_key='iteration', file_name='adversarial_loss.png', trigger=(100, 'iteration')))

    #test
    trainer.extend(ex.PlotReport(['gen/adv_loss' ],
        x_key='iteration', file_name='adv_gen_loss.png', trigger=(100, 'iteration')))

    trainer.extend(ex.PlotReport(['gen/ce_loss'],
        x_key='iteration', file_name='cross_entropy_loss.png', trigger=(100, 'iteration')))

    trainer.extend(ex.PlotReport(['gen/semi_st_loss'],
        x_key='iteration', file_name='self_teach_loss.png', trigger=(100, 'iteration')))

    trainer.extend(ex.PlotReport(['gen/loss', 'dis/loss', 'gen/semi_loss'],
        x_key='iteration', file_name='loss.png', trigger=(100, 'iteration')))

    trainer.extend(ex.PlotReport(['gen/loss', 'dis/loss', 'gen/semi_loss'],
        x_key='epoch', file_name='loss_details.png', trigger=(5, 'epoch')))

    trainer.extend(ex.PlotReport(['gen/semi_loss'],
        x_key='epoch', file_name='semi_loss.png', trigger=(1, 'epoch')))

    #snap
    trainer.extend(ex.snapshot_object(gen, 'gen_snapshot_epoch-{.updater.epoch}.npz'),
        trigger=(opt.snap_interval_epoch, 'epoch'))
    trainer.extend(ex.snapshot_object(dis, 'dis_snapshot_epoch-{.updater.epoch}.npz'),
        trigger=(opt.snap_interval_epoch, 'epoch'))

    trainer.extend(lambda *args: updater.save_img(),
        trigger=(opt.img_interval_iteration, 'iteration'), priority=PRIORITY_READER)

    trainer.extend(lambda *args: updater.ignition_semi_learning(),
        trigger=(opt.semi_ignit_iteration, 'iteration'), priority=PRIORITY_READER)

    trainer.extend(lambda *args: adam_lr_poly(opt, trainer), trigger=(100, 'iteration'))


    trainer.run() #start learning


if __name__ == '__main__':
    opt = get_options()

    train(opt)
