# coding: utf-8
import numpy as np
import lmdb
import cv2
import caffe

dir_main = '/home/jiange/dl_data/colorization/img_train_256x256/'  # image path
dir_txt = '/home/jiange/dl_data/colorization/test_txt_200.txt'  # 1
images = []

with open(dir_txt) as f:
    for line in f:
        fname,_ = line.split()
        images.append(dir_main+fname)

r = list(range(len(images)))
# creating lmdb dataset
count = 0
print 'Creating dataset...'
env = lmdb.open('/home/jiange/dl_data/colorization/test_lmdb_200_256x256', map_size=int(1e12))  # 2
for i in r:
    if (count+1) % 50 == 0:
        print 'Saving image: ', count+1
    X = cv2.imread(images[2]) # uint8
    X = np.transpose(X,(2,0,1)) # 改变坐标轴的顺序 3*w*h
    im_dat = caffe.io.array_to_datum(np.array(X).astype(np.uint8))
    str_id = '{:0>10d}'.format(count) # 10位 记录编号
    with env.begin(write=True) as txn:
        txn.put(str_id, im_dat.SerializeToString())
    count = count+1

env.close()