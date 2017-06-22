export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
#format: interval, learning_rate, total_iter, number_wokers, level, tuple_path, rl_model_path, r, eps_greedy 
./server_sarsa.bin 1 0.1 2000 2 16 /home/cgx/git_project/adaptive-system/input/word2vec/tuple_word2vec.pb /home/cgx/git_project/adaptive-system/reinforcement_learning_model/sarsa.pb 0.9 0.1 /home/cgx/git_project/adaptive-system/resources/
