export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
g++ -std=c++11 -I/home/cgx/git_project/adaptive-system/ -I/home/cgx/include/ -I/home/cgx/git_project/adaptive-system/tensorflow/ -I/home/cgx/git_project/adaptive-system/eigen-eigen  synchronous_client.cpp ../proto/*.cc -o client -L/home/cgx/git_project/adaptive-system/build/ -ltensorflow_cc -lalgorithms -L/usr/local/lib -lgrpc++ -lgrpc  -lgrpc++_reflection -L/home/cgx/lib/ -lprotobuf -lpthread -ldl
chmod a+x server
./server