export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/:/home/cgx/grpc/libs/opt/:/home/cgx/lib
#                 iter_num, worker_num, init_level, interval, start_level, end_level, eps_greedy, r, learning_rate, start_iter, predict_interval
./single_implement_sarsa.bin 5000 18 2 5 2 8 0.1 0.8 0.15 30 5.0 7.5 10
