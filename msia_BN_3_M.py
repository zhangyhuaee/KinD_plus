import tensorflow as tf
import tensorflow.contrib.slim as slim

def illu_attention_3_M(input_feature, input_i, name):
  kernel_size = 3
  kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
  with tf.variable_scope(name):
    concat = tf.layers.conv2d(input_i,
                              filters=1,
                              kernel_size=[kernel_size,kernel_size],
                              strides=[1,1],
                              padding="same",
                              activation=None,
                              kernel_initializer=kernel_initializer,
                              use_bias=False,
                              name='conv')
    assert concat.get_shape()[-1] == 1
    concat = tf.sigmoid(concat, 'sigmoid')
    
  return input_feature * concat#, concat

def pool_upsamping_3_M(input_feature, level, training, name):
  if level == 1:
    with tf.variable_scope(name):
      pu_conv = slim.conv2d(input_feature, input_feature.get_shape()[-1], [3,3], 1, padding='SAME' ,scope='pu_conv')
      pu_conv = tf.layers.batch_normalization(pu_conv, training=training)
      pu_conv = tf.nn.relu(pu_conv)
      conv_up = pu_conv
  elif level == 2:
    with tf.variable_scope(name):
      pu_net = slim.max_pool2d(input_feature, [2,2], 2, padding='SAME', scope='pu_net')
      pu_conv = slim.conv2d(pu_net, input_feature.get_shape()[-1], [3,3], 1, padding='SAME' ,scope='pu_conv')
      pu_conv = tf.layers.batch_normalization(pu_conv, training=training)
      pu_conv = tf.nn.relu(pu_conv)
      conv_up = slim.conv2d_transpose(pu_conv, input_feature.get_shape()[-1], [2,2], 2, padding='SAME', scope='conv_up')
  elif level == 4:
    with tf.variable_scope(name):
      pu_net = slim.max_pool2d(input_feature, [4,4], 4, padding='SAME', scope='pu_net')
      pu_conv = slim.conv2d(pu_net, input_feature.get_shape()[-1], [1,1], 1, padding='SAME' ,scope='pu_conv')
      pu_conv = tf.layers.batch_normalization(pu_conv, training=training)
      pu_conv = tf.nn.relu(pu_conv)
      conv_up_1 = slim.conv2d_transpose(pu_conv, input_feature.get_shape()[-1], [2,2], 2, padding='SAME', scope='conv_up_1')
      conv_up = slim.conv2d_transpose(conv_up_1, input_feature.get_shape()[-1], [2,2], 2, padding='SAME', scope='conv_up')

  return conv_up

def Multi_Scale_Module_3_M(input_feature, training, name):
    
    Scale_1 = pool_upsamping_3_M(input_feature, 1, training, name=name+'pu1')
    Scale_2 = pool_upsamping_3_M(input_feature, 2, training, name=name+'pu2')
    Scale_4 = pool_upsamping_3_M(input_feature, 4, training, name=name+'pu4')
    
    res = tf.concat([input_feature, Scale_1, Scale_2, Scale_4], axis=3)
    multi_scale_feature = slim.conv2d(res, input_feature.shape[3], [1,1], 1, padding='SAME', scope=name+'multi_scale_feature')
    return multi_scale_feature

def msia_3_M(input_feature, input_i, name, training):
    spatial_attention_feature = illu_attention_3_M(input_feature, input_i, name)
    msia_feature = Multi_Scale_Module_3_M(spatial_attention_feature, training, name)
    return msia_feature


