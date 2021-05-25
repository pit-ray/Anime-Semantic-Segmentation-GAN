# coding: utf-8
from chainer import report
import chainer.functions as F
from chainer.backends import cuda


def dis_loss(opt, real_d, fake_d, observer=None):
    # adversarial loss
    adv_loss = 0
    real_loss = 0
    fake_loss = 0
    if opt.adv_loss_mode == 'bce':
        real_loss = F.mean(F.softplus(-real_d))
        fake_loss = F.mean(F.softplus(fake_d))

    if opt.adv_loss_mode == 'mse':
        xp = cuda.get_array_module(real_d.array)
        real_loss = F.mean_squared_error(real_d, xp.ones_like(real_d.array))
        fake_loss = F.mean_squared_error(fake_d, xp.zeros_like(fake_d.array))

    if opt.adv_loss_mode == 'hinge':
        real_loss = F.mean(F.relu(1.0 - real_d))
        fake_loss = F.mean(F.relu(1.0 + fake_d))

    adv_loss = (real_loss + fake_loss) * 0.5

    loss = adv_loss

    if observer is not None:
        report({'loss': loss,
                'adv_loss': adv_loss,
                'real_loss': real_loss,
                'fake_loss': fake_loss}, observer=observer)

    return loss


def gen_loss(opt, fake_d, real_g, fake_g, eps=1e-12, observer=None):
    # adversarial loss
    adv_loss = 0
    fake_loss = 0
    if opt.adv_loss_mode == 'bce':
        fake_loss = F.mean(F.softplus(-fake_d))

    if opt.adv_loss_mode == 'mse':
        xp = cuda.get_array_module(fake_d.array)
        fake_loss = F.mean_squared_error(fake_d, xp.ones_like(fake_d.array))

    if opt.adv_loss_mode == 'hinge':
        fake_loss = -F.mean(fake_d)

    adv_loss = fake_loss

    adv_loss *= opt.adv_coef

    # cross-entropy loss
    ce_loss = -F.mean(real_g * F.log(fake_g + eps))

    loss = adv_loss + ce_loss

    if observer is not None:
        report({'loss': loss,
                'adv_loss': adv_loss,
                'fake_loss': fake_loss,
                'ce_loss': ce_loss}, observer=observer)

    return loss


def gen_semi_loss(opt, unlabel_d, unlabel_g, eps=1e-12, observer=None):
    # semi-supervised loss
    # HW-filter
    confidence_mask = F.sigmoid(unlabel_d).array > opt.semi_threshold

    # C-filter (which does pixels belong to each class)
    class_num = unlabel_g.shape[1]
    xp = cuda.get_array_module(unlabel_g.array)

    predict_index = unlabel_g.array.argmax(axis=1)
    predict_mask = xp.eye(class_num,
        dtype=unlabel_g.dtype)[predict_index].transpose(0, 3, 1, 2)

    # CHW-filter
    ground_truth = confidence_mask * predict_mask

    st_loss = -F.mean(ground_truth * F.log(unlabel_g + eps))

    # adversarial loss
    adv_loss = 0
    fake_loss = 0
    if opt.adv_loss_mode == 'bce':
        fake_loss = F.mean(F.softplus(-unlabel_d))

    if opt.adv_loss_mode == 'mse':
        xp = cuda.get_array_module(unlabel_d.array)
        fake_loss = F.mean_squared_error(unlabel_d, xp.ones_like(unlabel_d.array))

    if opt.adv_loss_mode == 'hinge':
        fake_loss = -F.mean(unlabel_d)

    adv_loss = fake_loss

    # weight
    adv_loss *= opt.semi_adv_coef
    st_loss *= opt.semi_st_coef

    loss = adv_loss + st_loss

    if observer is not None:
        report({'semi_loss': loss,
                'semi_adv_loss': adv_loss,
                'semi_fake_loss': fake_loss,
                'semi_st_loss': st_loss}, observer=observer)

    return loss
