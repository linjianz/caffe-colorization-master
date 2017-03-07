GLOG_logtostderr=0 GLOG_log_dir=/home/jiange/dl_data/colorization/log/ \
./build/tools/caffe train \
-solver ./examples/colorization/solver.prototxt \
-weights /home/jiange/dl_model/colorization/init_v2.caffemodel \
-gpu 0