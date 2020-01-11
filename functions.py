#coding: utf-8
from chainer.backends import cuda

#This parameter dependent on RGB-combination of label
CLASS_COLOR = [[1, 0, 0], #eye
               [0, 1, 0], #face
               [0, 0, 1], #hair
               [1, 0, 1], #other
               [1, 1, 0]] #background
BACKGROUND_INDEX = 4

#the range of RGB is from zero to one.
def label2onehot(label, threshold=0.4, skip_bg=False, dtype='uint8'):
    label = label > threshold

    onehot = None
    xp = cuda.get_array_module(label)
    na = xp.newaxis

    for i in range(len(CLASS_COLOR)):
        if skip_bg and i == BACKGROUND_INDEX:
            continue

        detecter = xp.array(CLASS_COLOR[i], dtype=dtype)[:, na, na]
        detecter = detecter.repeat(label.shape[1], axis=1)
        detecter = detecter.repeat(label.shape[2], axis=2)

        mask = xp.sum(label == detecter,
            axis=0, keepdims=True, dtype=dtype) == 3

        if i == 0:
            onehot = mask
        else:
            onehot = xp.concatenate((onehot, mask), axis=0)

    return onehot


def onehot2label(onehot, skip_bg=False, dtype='uint8'):
    xp = cuda.get_array_module(onehot)
    if skip_bg:
        any_class = xp.sum(onehot, axis=0, keepdims=True)
        bg = any_class == 0

        onehot = xp.concatenate((onehot, bg), axis=0)

    which_class = xp.argmax(onehot, axis=0)
    mask = xp.eye(len(CLASS_COLOR), dtype=dtype)[which_class].transpose(2, 0, 1)

    detecter = xp.array(CLASS_COLOR, dtype=dtype)
    label = xp.einsum('ij,ikl->jkl', detecter, mask)

    return label


def adam_lr_poly(opt, trainer):
    epoch = trainer.updater.epoch_detail
    max_epoch = trainer.stop_trigger.period
    
    threshold_epoch = max_epoch * opt.lr_poly_train_period
    if epoch < threshold_epoch:
        return

    epoch -= threshold_epoch
    max_epoch -= threshold_epoch

    glr = opt.g_lr * (1 - epoch / max_epoch)**opt.lr_poly_power
    dlr = opt.d_lr * (1 - epoch / max_epoch)**opt.lr_poly_power

    trainer.updater.get_optimizer('gen').alpha = glr
    trainer.updater.get_optimizer('dis').alpha = dlr

    print('[extention:adam_lr_poly] generater optimizer learning rate is %.6f.' % glr)
    print('[extention:adam_lr_poly] discriminator optimizer learning rate is %.6f.' % dlr)