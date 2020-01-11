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
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./LOLdataset/eval15/', help='directory for testing inputs')

args = parser.parse_args()

sess = tf.Session()
training = tf.placeholder_with_default(False, shape=(), name='training')
input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')
input_low_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_r')
input_low_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
input_low_i_ratio = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i_ratio')

[R_decom, I_decom] = DecomNet(input_decom)
decom_output_R = R_decom
decom_output_I = I_decom
output_r = Restoration_net(input_low_r, input_low_i, training)
output_i = Illumination_adjust_net(input_low_i, input_low_i_ratio)

# load pretrained model parameters
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
    print('No decomnet pretrained model!')

checkpoint_dir_adjust = './checkpoint/illu_model/'
ckpt_adjust=tf.train.get_checkpoint_state(checkpoint_dir_adjust)
if ckpt_adjust:
    print('loaded '+ckpt_adjust.model_checkpoint_path)
    saver_adjust.restore(sess,ckpt_adjust.model_checkpoint_path)
else:
    print("No adjust net pretrained model!")

checkpoint_dir_restoration = './checkpoint/restoration_model/'
ckpt=tf.train.get_checkpoint_state(checkpoint_dir_restoration)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restoration.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No restoration net pretrained model!")

###load eval data
eval_low_data = []
eval_img_name =[]
eval_low_data_name = glob(args.test_dir + '/low/*.png')
eval_low_data_name.sort()
for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name.append(name)
    eval_low_im = load_images(eval_low_data_name[idx])
    eval_low_data.append(eval_low_im)
    print(eval_low_im.shape)
# To get better results, the illumination adjustment ratio is computed based on the decom_i_high, so we also need the high data.
eval_high_data = []
eval_high_data_name = glob(args.test_dir + '/high/*.png')
eval_high_data_name.sort()
for idx in range(len(eval_high_data_name)):
    eval_high_im = load_images(eval_high_data_name[idx])
    eval_high_data.append(eval_high_im)

sample_dir = args.save_dir +'/LOLdataset/'
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

print("Start evalating!")
start_time = time.time()
for idx in range(len(eval_low_data)):
    print(idx)
    name = eval_img_name[idx]
    print('Evaluate image %s'%name)
    input_low = eval_low_data[idx]
    input_low_eval = np.expand_dims(input_low, axis=0)
    input_high = eval_high_data[idx]
    input_high_eval = np.expand_dims(input_high, axis=0)
    h, w, _ = input_low.shape

    decom_r_low, decom_i_low = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_low_eval})
    decom_r_high, decom_i_high = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_high_eval})
    
    restoration_r = sess.run(output_r, feed_dict={input_low_r: decom_r_low, input_low_i: decom_i_low})

    ratio = np.mean(((decom_i_high))/(decom_i_low+0.0001))
    ratio2 = np.mean(((decom_r_high))/(restoration_r+0.0001))
    if ratio2<1.1:
        i_low_data_ratio = np.ones([h, w])*(ratio)
    else:
        i_low_data_ratio = np.ones([h, w])*(ratio+ratio2)
    
    i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis =2)
    i_low_ratio_expand2 = np.expand_dims(i_low_ratio_expand, axis=0)

    adjust_i = sess.run(output_i, feed_dict={input_low_i: decom_i_low, input_low_i_ratio: i_low_ratio_expand2})
    fusion = restoration_r*adjust_i
    
    save_images(os.path.join(sample_dir, '%s_kindle_v2.png' % (name)), fusion)
    #save_images(os.path.join(sample_dir, '%s_decom_i_low.png' % (name)), decom_i_low)
    #save_images(os.path.join(sample_dir, '%s_adjust_i_%f.png' % (name, (ratio+ratio2)) ), adjust_i)
    #save_images(os.path.join(sample_dir, '%s_denoise_r.png' % (name)), restoration_r)
