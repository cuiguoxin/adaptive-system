export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
#format: interval, learning_rate, total_iter, number_wokers, const_level, tuple_path, rl_model_path, r, eps_greedy, material_path 
./server_baseline_combine.bin 1 0.1 20000 6 /home/cgx/git_project/adaptive-system/input/cifar10/tuple_gradient_descent.pb /home/cgx/git_project/adaptive-system/resources/cifar-10-batches-bin/test_batch.bin /home/cgx/git_project/adaptive-system/input/cifar10/preprocess.pb /home/cgx/git_project/adaptive-system/input/cifar10/tuple_adam_predict.pb 105000 2 1400 4

