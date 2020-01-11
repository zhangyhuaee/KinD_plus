# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
from skimage import color,filters
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results/', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./test_images/', help='directory for testing inputs')
parser.add_argument('--adjustment', dest='adjustment', default=False, help='whether to adjust illumination')
parser.add_argument('--ratio', dest='ratio', default=5.0, help='ratio for illumination adjustment')

args = parser.parse_args()

sess = tf.Session()
training = tf.placeholder_with_default(False, shape=(), name='training')
input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')
input_low_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_r')
input_low_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
input_high_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high_r')
input_high_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_high_i')
input_low_i_ratio = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i_ratio')

[R_decom, I_decom] = DecomNet(input_decom)
decom_output_R = R_decom
decom_output_I = I_decom
output_r = Restoration_net(input_low_r, input_low_i, training)
output_i = Illumination_adjust_net(input_low_i, input_low_i_ratio)

# load pretrained model
var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_adjust = [var for var in tf.trainable_variables() if 'I_enhance_Net' in var.name]
var_restoration = [var for var in tf.trainable_variables() if 'Denoise_Net' in var.name]
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_restoration += bn_moving_vars

saver_Decom = tf.train.Saver(var_list = var_Decom)
saver_adjust = tf.train.Saver(var_list=var_adjust)
saver_restoration = tf.train.Saver(var_list=var_restoration)

decom_checkpoint_dir ='./checkpoint/decom_model/'
ckpt_pre=tf.train.get_checkpoint_state(decom_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No decomnet checkpoint!')

checkpoint_dir_adjust = './checkpoint/illu_model/'
ckpt_adjust=tf.train.get_checkpoint_state(checkpoint_dir_adjust)
if ckpt_adjust:
    print('loaded '+ckpt_adjust.model_checkpoint_path)
    saver_adjust.restore(sess,ckpt_adjust.model_checkpoint_path)
else:
    print("No adjust pre model!")

checkpoint_dir_restoration = './checkpoint/restoration_model/'
ckpt=tf.train.get_checkpoint_state(checkpoint_dir_restoration)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restoration.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No restoration pre model!")

###load eval data
eval_low_data = []
eval_img_name =[]
eval_low_data_name = glob(args.test_dir+'*')
eval_low_data_name.sort()
for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name.append(name)
    eval_low_im = load_images(eval_low_data_name[idx])
    print(eval_low_im.shape)
    h,w,c = eval_low_im.shape
# the size of test image H and W need to be multiple of 4, if it is not a multiple of 4, we will discard some border pixels.  
    h_tmp = h%4
    w_tmp = w%4
    eval_low_im_resize = eval_low_im[0:h-h_tmp, 0:w-w_tmp, :]
    print(eval_low_im_resize.shape)
    eval_low_data.append(eval_low_im_resize)

sample_dir = args.save_dir 
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

print("Start evalating!")
start_time = time.time()
for idx in range(len(eval_low_data)):
    print(idx)
    name = eval_img_name[idx]
    input_low = eval_low_data[idx]
    input_low_eval = np.expand_dims(input_low, axis=0)
    h, w, _ = input_low.shape

    decom_r_low, decom_i_low = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_low_eval})
    restoration_r = sess.run(output_r, feed_dict={input_low_r: decom_r_low, input_low_i: decom_i_low, training: False})
### change the ratio to get different exposure level, the value can be 0-5.0
    ratio = float(args.ratio)
    i_low_data_ratio = np.ones([h, w])*(ratio)
    i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis =2)
    i_low_ratio_expand2 = np.expand_dims(i_low_ratio_expand, axis=0)
    adjust_i = sess.run(output_i, feed_dict={input_low_i: decom_i_low, input_low_i_ratio: i_low_ratio_expand2})

#The restoration result can find more details from very dark regions, however, it will restore the very dark regions
#with gray colors, we use the following operator to alleviate this weakness.  
    decom_r_sq = np.squeeze(decom_r_low)
    r_gray = color.rgb2gray(decom_r_sq)
    r_gray_gaussion = filters.gaussian(r_gray, 3)
    low_i =  np.minimum((r_gray_gaussion*2)**0.5,1)
    low_i_expand_0 = np.expand_dims(low_i, axis = 0)
    low_i_expand_3 = np.expand_dims(low_i_expand_0, axis = 3)
    result_denoise = restoration_r*low_i_expand_3
    fusion4 = result_denoise*adjust_i

    if args.adjustment:
        fusion = decom_i_low*input_low_eval + (1-decom_i_low)*fusion4
    else:
        fusion = decom_i_low*input_low_eval + (1-decom_i_low)*result_denoise
    #fusion2 = decom_i_low*input_low_eval + (1-decom_i_low)*restoration_r
    save_images(os.path.join(sample_dir, '%s_KinD_plus.png' % (name)), fusion)
    
    
