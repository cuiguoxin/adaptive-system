export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
#format: interval, learning_rate, total_iter, number_wokers, level, tuple_path, rl_model_path, r, eps_greedy 
./server.bin 3 0.1 3000 2 8 /home/cgx/git_project/adaptive-system/input/cifar10/tuple_adam.pb /home/cgx/git_project/adaptive-system/reinforcement_learning_model/sarsa.pb 0.9 0.1
