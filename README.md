# Colorful Image Colorization

This is a colorization project from[Project Page](http://richzhang.github.io/colorization/)

The train and test programs are in ./examples/colorization/

## test

run the command `$ python examples/colorization/colorization.py`, first of all, you have to change the directory and models in the colorization.py.

## train

**1. create lmdb**

`$ ./examples/colorization/create_lmdb_caffe.sh`

**2. start training**

`$ ./examples/colorization/train_model.sh`, if you want to resume from a interrupted models, just run `$ ./examples/colorization/train_resume.sh`