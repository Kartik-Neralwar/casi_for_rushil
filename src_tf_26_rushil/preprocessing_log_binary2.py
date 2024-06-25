"""
Contains methods which prepare data so that it may be fed into predictive
models, as well as multiple helper functions which handle loading the data and
other similar tasks.
"""
import itertools

import re
from pathlib import Path
import random
import numpy as np
from astropy.io import fits
import os
import time


rand_pos = 0 # number of files to pop from positive training set
rand_neg = 0 # number of files to pop from negative training set

feedback = 'wind'

random.seed(9)

def density_preprocessing(x, y):
    x = normalize(x)
    y = normalize(y)

    x = np.sign(x) * np.log10(1 + np.abs(x))
    x /= np.std(x)
    y = np.sign(y) * np.log10(1 + np.abs(y))
    y /= np.std(y)

    x = slice_density(x)
    y = slice_density(y)

    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    return x, y

def co_preprocessing(data_path, raw_path, neg_data_path):
    tracer_files_train = get_co_tracer_files(data_path)
    _ = [tracer_files_train.pop(random.randrange(len(tracer_files_train))) for _ in itertools.repeat(None, rand_pos)]
    tracer_files_neg_train = get_co_tracer_files(neg_data_path)
    _ = [tracer_files_neg_train.pop(random.randrange(len(tracer_files_neg_train))) for _ in itertools.repeat(None, rand_neg)]

    tracer_files = tracer_files_train + tracer_files_neg_train
     
    co_files = get_co_files(data_path, raw_path, tracer_files)

    co = load_fits(co_files)

    x = np.asarray(co)
    del co
    tracer = load_fits(tracer_files)
    y = np.abs(np.asarray(tracer))

    # x[x < 0] = 0
    # indx_mask=(y>0.001)
    # y[~indx_mask]=0
    # y[indx_mask]=y[indx_mask]*5+x[indx_mask]
    # indx_mask=(y<0.7)
    # y[indx_mask]=0
#        indx_mask=(y>0.4)
#        y[indx_mask]=1


    min_data=np.min(x)
    x = np.log(x - min_data + 1.)
    y = np.log(y - min_data + 1.)

    mean_data=np.mean(x)
    std_data=np.std(x)

    if std_data ==0:
        std_data=1
    
    noise_cut = mean_data + (3*std_data)
    
    mean_data = np.mean(x[x>noise_cut])
    std_data = np.std(x[x>noise_cut])
    
    if std_data ==0:
        std_data=1

    # y=y-mean_data
    # y=y/std_data
    x=x-mean_data
    x=x/std_data


    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)

    return x, y

#updated the function get_co_files
def get_co_files(data_path, raw_path, tracer_files):    
    raw_files_1 = [x.replace('_neg_', '_') for x in tracer_files]
    raw_files = [x.replace(data_path, raw_path) for x in raw_files_1]
    return [x.replace(feedback, 'raw') for x in raw_files]

def get_co_tracer_neg_files(neg_data_path):
    file_keyword = r'.*' + feedback+ r'.*'
    return [str(x) for x in Path(neg_data_path).glob('*.fits')
            if re.match(file_keyword, str(x))]

## updated by me # this ensures that even if there are other feedback tracers in this folder by mistake, they don't count 
def get_co_tracer_files(data_path):
    file_keyword = r'.*' + feedback+ r'.*'
    return [str(x) for x in Path(data_path).glob('*.fits')
            if re.match(file_keyword, str(x))]

def load_fits(files):
    output_data = []

    for file in files:
        with fits.open(file) as fits_data:
            output_data.append(fits_data[0].data)
            # fits_data.close()
            # del fits_data
    return output_data


def prediction_to_fits(pred, ref_files=None, outpath='../models/prediction.fits'):
    with fits.open(ref_files[0]) as fits_ref:
        ref_shape = fits_ref[0].data.shape
        fits_ref[0].data[:] = np.squeeze(pred)[:ref_shape[0], :ref_shape[1], :ref_shape[2]]
        fits_ref.writeto(outpath, overwrite=True)
        # fits_ref.close()


#def pad_data(data, value=0.):
#    return np.pad(data,
#                  ((0, 0), (0, 1), (0, 1)),
#                  'constant',
#                  constant_values=value)


def slice_density(data):
    slices = np.empty((np.sum(data.shape), data.shape[1], data.shape[2]))
    slice_count = 0

    for i in range(data.shape[0]):
        slices[slice_count, :] = data[i, :, :]
        slice_count += 1

    for j in range(data.shape[1]):
        slices[slice_count, :] = data[:, j, :]
        slice_count += 1

    for k in range(data.shape[2]):
        slices[slice_count, :] = data[:, :, k]
        slice_count += 1

    return slices


def normalize(data):
    data -= np.mean(data)
    if np.std(data) ==0:
        data=data
    else:
        data /= np.std(data)

    return data