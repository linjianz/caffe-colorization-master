GLOG_logtostderr=0 GLOG_log_dir=/home/jiange/dl_data/colorization/log/ \
./build/tools/caffe train \
-solver ./examples/colorization/solver.prototxt \
-snapshot /home/jiange/dl_model/colorization/20170221_1/colorization_iter_20000.solverstate \
-gpu 0
