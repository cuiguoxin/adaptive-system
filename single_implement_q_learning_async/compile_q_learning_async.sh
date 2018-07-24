export LD_LIBRARY_PATH=~/git_project/adaptive-system/build/
BASE_PATH=~/git_project/adaptive-system
g++ -std=c++11 -I$BASE_PATH -I/home/cgx/include/ -I$BASE_PATH/tensorflow/ -I$BASE_PATH/eigen-eigen async_implementation_q_learning.cc  ../quantization/util/*.cc ../proto/*.cc ../server/q_learning.cc ../server/reward.cc -o single_async_implement_q_learning.bin -L$BASE_PATH/build -ltensorflow_cc -ltensorflow_framework  -L/usr/local/lib -lgrpc++ -lgrpc  -lgrpc++_reflection -L/home/cgx/lib/ -lprotobuf -lpthread -ldl 2>error.log
chmod a+x single_async_implement_q_learning.bin
