export LD_LIBRARY_PATH=~/git_project/adaptive-system/build/:/home/cgx/grpc/libs/opt/:/home/cgx/lib
#                 total_iter_num, worker_num, level, learning_rate, start_parallel, tuple_pb_path
./single_async_implement_baseline.bin 5000 8 8 0.05 20 128 ~/git_project/adaptive-system/input/cifar10/tuple_gradient_descent.pb 5
