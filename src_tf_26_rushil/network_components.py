"""
Contains functions which construct basic building blocks for Keras models.
"""


import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv3D
from tensorflow.keras.losses import mae, mse

#from tf_ssim import ssim_multiscale


def soft_iou_loss(y_true, y_pred):
    """
    A loss function based on the Intersection Over Union score, implemented
    using the Keras backend.

    Args:
        y_true: symbolic tensor representing the target values.
        y_pred: symbolic tensor representing the predicted values.

    Returns:
        symbolic tensor containing the IOU loss for each sample.
    """
    intersection = K.sum(K.batch_flatten(y_true * y_pred), axis=-1)
    union = K.sum(K.batch_flatten(K.maximum(y_true, y_pred)), axis=-1)
    return 1 - (intersection / union)


#def mixed_msssim_loss(y_true, y_pred, alpha=0.84, max_val=255.):
#    return alpha * (1. - ssim_multiscale(y_true, y_pred, max_val)) + \
#           (1. - alpha) * K.mean(mae(y_true, y_pred), axis=[-1, -2])


def mixed_l1_l2_loss(y_true, y_pred, alpha=0.5):
    return alpha * mse(y_true, y_pred) + (1 - alpha) * mae(y_true, y_pred)


def weighted_mse(y_true, y_pred):
    weights = y_true / K.max(y_true) + 1.
    weights /= K.sum(weights)
    return K.squeeze(weights, axis=-1) * mse(y_true, y_pred)


def res_block(x,
              filters=16,
              filter_shape=(3, 3, 3),
              activation='selu',
              merge=Add(),
              project=False):
    """
    Implements the two convolution residual block described in
        https://arxiv.org/pdf/1512.03385.pdf

    Args:
        x: Input tensor
        filters: Number of filters used in convolutions
        filter_shape: Dimensions of the convolution filters
        activation: (str) Usually 'relu', 'elu', or 'selu'
        merge: Layer used to merge the skip connection, Concatenate or Add
        project: Apply a 1x1 convolution to the skip connection

    Returns:
        Symbolic output tensor for the final activation
    """
    pred = Conv3D(filters, filter_shape, padding='same')(x)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)

    pred = Conv3D(filters, filter_shape, padding='same')(pred)
    pred = BatchNormalization()(pred)

    if project:
        x = Conv3D(filters, (1, 1, 1))(x)
        x = BatchNormalization()(x)
    pred = merge([x, pred])
    return Activation(activation)(pred)


def res_bottlneck(x,
                  filters=16,
                  neck_filters=None,
                  filter_shape=(3, 3, 3),
                  activation='selu',
                  merge=Add(),
                  project=False):
    """
    Implements the three convolution bottleneck residual block described in
        https://arxiv.org/pdf/1512.03385.pdf

    Args:
        x: Input tensor
        filters: Number of filters used in output convolution
        neck_filters: Number of filters used in bottleneck convolutions
        filter_shape: Dimensions of the convolution filters
        activation: (str) Usually 'relu', 'elu', or 'selu'
        merge: Layer used to merge the skip connection, Concatenate or Add
        project: Apply a 1x1 convolution to the skip connection

    Returns:
        Symbolic output tensor for the final activation
    """
    if not neck_filters:
        neck_filters = max(filters // 4, 1)

    pred = Conv3D(neck_filters, (1,1,1))(x)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)

    pred = Conv3D(neck_filters, filter_shape, padding='same')(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)

    pred = Conv3D(filters, (1, 1, 1))(pred)
    pred = BatchNormalization()(pred)

    if project:
        x = Conv3D(filters, (1, 1, 1))(x)
        x = BatchNormalization()(x)

    pred = merge([x, pred])

    return Activation(activation)(pred)


def dilated_residual_block(
        x,
        filters=16,
        dilations=tuple(2 ** x for x in range(4)),
        activation='selu',
        project=False,
        merge=Add()):
    pred = dilated_block(
        x,
        filters=filters,
        dilations=dilations,
        activation=activation,
        merge=merge)
    pred = dilated_block(
        pred,
        filters=filters,
        dilations=dilations,
        activation=activation,
        merge=merge)

    if project:
        x = Conv3D(filters, 1)(x)
        x = BatchNormalization()(x)
    return merge([pred, x])


def dilated_block(
        x,
        filters=16,
        dilations=tuple(2 ** x for x in range(4)),
        activation='selu',
        merge=Add()):
    pred = BatchNormalization()(x)
    pred = Activation(activation)(pred)
    preds = [Conv3D(filters, (3, 3, 3), dilation_rate=d, padding='same')(pred) for d in dilations]
    preds = [BatchNormalization()(p) for p in preds]
    return merge(preds)


def cyclic_lr_schedule(lr0=0.2, total_steps=400, cycles=8):
    def lr_schedule(step, lr=0.):
        return 0.5 * lr0 * (np.cos(np.pi * (step % np.ceil(total_steps / cycles)) / np.ceil(total_steps / cycles)) + 1)
    return lr_schedule


class CyclicLRScheduler(Callback):
    """Cyclic learning rate scheduler.
    Args:
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: (int) 0 -> quiet, 1 -> update messages.
    """
    def __init__(self, schedule=None, verbose=0):
        super().__init__()
        self.schedule = cyclic_lr_schedule() if schedule is None else schedule
        self.verbose = verbose
        self.step = 0

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = self.schedule(self.step,
                               lr=float(K.get_value(self.model.optimizer.lr)))
        except TypeError:  # compatibility with old API
            lr = self.schedule(self.step)

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')

        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            with open('lr_schedule.log', 'a') as f:
                print(f'\nStep {self.step}: learning rate = {lr}.', file=f)
        self.step += 1


def dense_block():
    """
    To be implemented, see https://arxiv.org/abs/1608.06993
    """
    pass


def sparse_block():
    """
    To be implemented, see https://arxiv.org/abs/1801.05895
    """
    pass
