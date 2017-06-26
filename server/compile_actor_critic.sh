export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
g++ -std=c++11 -I/home/cgx/git_project/adaptive-system/ -I/home/cgx/include/ -I/home/cgx/git_project/adaptive-system/tensorflow/ -I/home/cgx/git_project/adaptive-system/eigen-eigen  synchronous_server_using_actor_critic.cc actor_critic.cc reward.cc indexed_slices.cc ../proto/*.cc -o server_actor_critic.bin -L/home/cgx/git_project/adaptive-system/build/ -ltensorflow_cc -lalgorithms -L/usr/local/lib -lgrpc++ -lgrpc  -lgrpc++_reflection -L/home/cgx/lib/ -lprotobuf -lpthread -ldl
chmod a+x server_actor_critic.bin
