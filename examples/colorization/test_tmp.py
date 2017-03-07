import numpy as np
import matplotlib.pyplot as plt
import skimage.color as color
import scipy.ndimage.interpolation as sni
import caffe

dir_model = '/home/jiange/dl_model/colorization/colorization_release_v2_norebal.caffemodel'  # richard

dir_main = '/home/jiange/dl_data/colorization/img_input/'
dir_in = '0_input.jpg'
dir_out = '0_output_richard.jpg'

# matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)

gpu_id = 0
caffe.set_mode_gpu()
caffe.set_device(gpu_id)
# Select desired model
net = caffe.Net('./colorization_deploy_v2.prototxt', dir_model, caffe.TEST)
# If you are training your own network, you may replace the *.caffemodel path with your trained network.
(H_in,W_in) = net.blobs['data_l'].data.shape[2:]  # get input shape
(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:]  # get output shape
print H_in, W_in, H_out, W_out

pts_in_hull = np.load('/home/jiange/dl/colorization/resources/pts_in_hull.npy') # load cluster centers
net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel

# load the original image
img_rgb = caffe.io.load_image(dir_main + dir_in)  # if input image is color, then output [h, w, 3]; else copy [h, w] 3 times into [h, w, 3]


img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
img_l = img_lab[:,:,0] # pull out L channel
(H_orig,W_orig) = img_rgb.shape[:2] # original image size

# create grayscale version of image (just for displaying)
img_lab_bw = img_lab.copy()
img_lab_bw[:,:,1:] = 0
img_rgb_bw = color.lab2rgb(img_lab_bw)

# resize image to network input size
img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
img_lab_rs = color.rgb2lab(img_rs)
img_l_rs = img_lab_rs[:,:,0]

# show original image, along with grayscale input to the network
img_pad = np.ones((H_orig,W_orig/10,3))

plt.imshow(np.hstack((img_rgb, img_pad, img_rgb_bw)))
plt.title('(Left) Loaded image   /   (Right) Grayscale input to network')
plt.axis('off')

net.blobs['data_l'].data[0,0,:,:] = img_l_rs - 50  # subtract 50 for mean-centering

net.forward()  # run network
ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0))  # this is our result
ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1))  # upsample to match size of original image L
img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2)  # concatenate with original image L
print "lab2rgb..."
img_rgb_out = np.clip(color.lab2rgb(img_lab_out),0,1)  # convert back to rgb
print "save image..."
plt.imsave(dir_main + dir_out, img_rgb_out)
plt.imshow(img_rgb_out)
plt.axis('off')

