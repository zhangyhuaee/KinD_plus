import numpy as np
from PIL import Image
import tensorflow as tf
import scipy.stats as st
from skimage import io,data,color
from functools import reduce

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def gradient_no_abs(input_tensor, direction):
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    gradient_orig = tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME')
    grad_min = tf.reduce_min(gradient_orig)
    grad_max = tf.reduce_max(gradient_orig)
    grad_norm = tf.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm

def gradient(input_tensor, direction):
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    gradient_orig = tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
    grad_min = tf.reduce_min(gradient_orig)
    grad_max = tf.reduce_max(gradient_orig)
    grad_norm = tf.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

def tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_images(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm

def load_images_no_norm(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    
    return img

def bright_channel_2(input_img):
    h, w = input_img.shape[:2]  
    I = input_img  
    res = np.minimum(I  , I[[0]+range(h-1)  , :])  
    res = np.minimum(res, I[range(1,h)+[h-1], :])  
    I = res  
    res = np.minimum(I  , I[:, [0]+range(w-1)])  
    res = np.minimum(res, I[:, range(1,w)+[w-1]])
    return res  

def bright_channel(input_img):
    r = input_img[:,:,0]
    g = input_img[:,:,1]
    b = input_img[:,:,2]
    m,n = r.shape
    print(m,n)
    tmp = np.zeros((m,n))
    b_c = np.zeros((m,n))
    for i in range(0,m-1):
        for j in range(0,n-1):

            tmp[i,j] = np.max([r[i,j], g[i,j]])
            b_c[i,j] = np.max([tmp[i,j], b[i,j]])
    return b_c



def load_raw_high_images(file):
    raw = rawpy.imread(file)
    im_raw = raw.postprocess(use_camera_wb = True, half_size = False, no_auto_bright=True, output_bps=16)
    #im_raw = np.maximum(im_raw - 512,0)/ (65535 - 512)
    im_raw = np.float32(im_raw/65535.0)
    im_raw_min = np.min(im_raw)
    im_raw_max = np.max(im_raw)
    a_weight = np.float32(im_raw_max - im_raw_min)
    im_norm = np.float32((im_raw - im_raw_min) / a_weight)
    return im_norm, a_weight

def load_raw_images(file):
    raw = rawpy.imread(file)
    im_raw = raw.postprocess(use_camera_wb = True, half_size = False, no_auto_bright=True, output_bps=16)
    #im_raw = np.maximum(im_raw - 512,0)/ (65535 - 512)
    im_raw = np.float32(im_raw/65535.0)
    im_raw_min = np.min(im_raw)
    im_raw_max = np.max(im_raw)
    a_weight = np.float32(im_raw_max - im_raw_min)
    im_norm = np.float32((im_raw - im_raw_min) / a_weight)
    return im_norm, a_weight

def load_raw_low_images(file):
    raw = rawpy.imread(file)
    im_raw = raw.postprocess(use_camera_wb = True, half_size = False, no_auto_bright=True, output_bps=16)
    im_raw = np.maximum(im_raw - 512.0,0)/ (65535.0 - 512.0)
    im_raw = np.float32(im_raw)
    im_raw_min = np.min(im_raw)
    print(im_raw_min)
    im_raw_max = np.max(im_raw)
    print(im_raw_max)
    a_weight = np.float32(im_raw_max - im_raw_min)
    im_norm = np.float32((im_raw - im_raw_min) / a_weight)
    print(a_weight)
    return im_norm, a_weight

def load_images_and_norm(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    norm_coeff = np.float32(img_max - img_min)
    return img_norm, norm_coeff

def load_images_and_a_and_norm(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    a_weight = np.float32(img_max - img_min)
    return img, img_norm, a_weight

def load_images_and_a_003(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    img_norm = (np.maximum(img_norm, 0.03)-0.03) / 0.97
    a_weight = np.float32(img_max - img_min)
    return img_norm, a_weight


def load_images_no_norm(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0


def load_images_uint16(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 65535.0

def load_images_hsv(file):
    im = io.imread(file)
    hsv = color.rgb2hsv(im)

    return hsv

def save_images(filepath, result_1, result_2 = None, result_3 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    result_3 = np.squeeze(result_3)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)
    if not result_3.any():
        cat_image = cat_image
    else:
        cat_image = np.concatenate([cat_image, result_3], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')
