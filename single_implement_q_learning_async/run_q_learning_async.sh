export LD_LIBRARY_PATH=~/git_project/adaptive-system/build/:/home/cgx/grpc/libs/opt/:/home/cgx/lib
#                 total_iter_num, worker_num, level, learning_rate, start_parallel, tuple_pb_path, interval_to_exchange_variable:5, q_learning_interval:20, q_learning_r:0.8, q_learning_eps_greedy:0.1, q_learning_start_level:2, q_learning_end_level:8, q_learning_init_level:2, computing_time:0.1, one_bit_time:0.15
nohup ./single_async_implement_q_learning.bin 2500 8 8 0.05 20 128 ~/git_project/adaptive-system/input/cifar10/tuple_gradient_descent.pb 5 20 0.8 0.1 2 8 2 0.1, 0.15 &
sleep 1s
tail -f nohup.out