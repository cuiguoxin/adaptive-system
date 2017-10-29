export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
#format: interval, learning_rate, total_iter, number_wokers, init_level, tuple_path, rl_model_path, r, eps_greedy, material_path 
./server_sarsa.bin 4 0.1 2000 6 1 /home/cgx/git_project/adaptive-system/input/cifar10/tuple_adam.pb /home/cgx/git_project/adaptive-system/reinforcement_learning_model/sarsa_continous.pb 105000 4
