import numpy as np
import skimage.io as sio
import skimage.color as color
import scipy.ndimage.interpolation as sni
import caffe
from tqdm import tqdm
import warnings


def colorize_with_richard():
    dir_model = '/home/jiange/dl_model/colorization/colorization_release_v2_norebal.caffemodel'  # richard
    dir_input = '/home/jiange/dl_data/colorization/img_test_500/'
    dir_output = '/home/jiange/dl_data/colorization/img_output_richard/'
    gpu_id = 0
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net('./colorization_deploy_v2.prototxt', dir_model, caffe.TEST)
    (H_in, W_in) = net.blobs['data_l'].data.shape[2:]  # get input shape
    (H_out, W_out) = net.blobs['class8_ab'].data.shape[2:]  # get output shape
    pts_in_hull = np.load('/home/jiange/dl/colorization/resources/pts_in_hull.npy')  # load cluster centers
    # populate cluster centers as 1x1 convolution kernel
    net.params['class8_ab'][0].data[:, :, 0, 0] = pts_in_hull.transpose((1, 0))

    for i in tqdm(range(501)):
        name = str(i)
        # load the original image
        # if input image is color, then load rgb with shape[h, w, 3]
        # else copy gray image [h, w] 3 times into [h, w, 3]
        img_rgb = caffe.io.load_image(dir_input + name + '_input.jpg')
        img_lab = color.rgb2lab(img_rgb)  # convert image to lab color space
        img_l = img_lab[:, :, 0]  # pull out L channel
        (H_orig, W_orig) = img_rgb.shape[:2]  # original image size

        # resize image to network input size
        img_rs = caffe.io.resize_image(img_rgb, (H_in, W_in))
        img_lab_rs = color.rgb2lab(img_rs)
        img_l_rs = img_lab_rs[:, :, 0]

        net.blobs['data_l'].data[0, 0, :, :] = img_l_rs - 53  # subtract 50 for mean-centering

        net.forward()  # run network
        ab_dec = net.blobs['class8_ab'].data[0, :, :, :].transpose((1, 2, 0))  # this is our result
        ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out, 1.*W_orig/W_out, 1))  # upsample to match size of original image L
        img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)  # concatenate with original image L
        img_rgb_out = np.clip(color.lab2rgb(img_lab_out), 0, 1)  # convert back to rgb
        sio.imsave(dir_output + name + '_output_richard.jpg', img_rgb_out)


def colorize_with_jiange():
    # dir_model = '/home/jiange/dl_model/colorization/fine_tuning_model/colorization_iter_50000.caffemodel'
    # dir_model = '/home/jiange/dl_model/colorization/colorization_release_v2_norebal.caffemodel'
    dir_model = '/home/jiange/dl_model/colorization/20170224/colorization_iter_49000.caffemodel'

    dir_input = '/home/jiange/dl_data/colorization/img_test_500/'
    dir_output = '/home/jiange/dl_data/colorization/img_output_jiange_49k/'

    gpu_id = 0
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net('./colorization_deploy_v2.prototxt', dir_model, caffe.TEST)
    (H_in, W_in) = net.blobs['data_l'].data.shape[2:]  # get input shape
    (H_out, W_out) = net.blobs['class8_ab'].data.shape[2:]  # get output shape
    pts_in_hull = np.load('/home/jiange/dl/colorization/resources/pts_in_hull.npy')  # load cluster centers
    # populate cluster centers as 1x1 convolution kernel
    net.params['class8_ab'][0].data[:, :, 0, 0] = pts_in_hull.transpose((1, 0))

    for i in tqdm(range(501)):
        name = str(i)
        # load the original image
        # if input image is color, then load rgb with shape[h, w, 3]
        # else copy gray image [h, w] 3 times into [h, w, 3]
        img_rgb = caffe.io.load_image(dir_input + name + '_input.jpg')
        img_lab = color.rgb2lab(img_rgb)  # convert image to lab color space
        img_l = img_lab[:, :, 0]  # pull out L channel
        (H_orig, W_orig) = img_rgb.shape[:2]  # original image size

        # resize image to network input size
        img_rs = caffe.io.resize_image(img_rgb, (H_in, W_in))
        img_lab_rs = color.rgb2lab(img_rs)
        img_l_rs = img_lab_rs[:, :, 0]

        net.blobs['data_l'].data[0, 0, :, :] = img_l_rs - 53  # subtract 50 for mean-centering

        net.forward()  # run network
        ab_dec = net.blobs['class8_ab'].data[0, :, :, :].transpose((1, 2, 0))  # this is our result
        ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out, 1.*W_orig/W_out, 1))  # upsample to match size of original image L
        img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)  # concatenate with original image L
        img_rgb_out = np.clip(color.lab2rgb(img_lab_out), 0, 1)  # convert back to rgb
        sio.imsave(dir_output + name + '_7.jpg', img_rgb_out)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    colorize_with_jiange()
    # colorize_with_richard()
