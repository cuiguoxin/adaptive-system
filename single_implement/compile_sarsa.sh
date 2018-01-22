export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
g++ -std=c++11 -I/home/cgx/git_project/adaptive-system/ -I/home/cgx/include/ -I/home/cgx/git_project/adaptive-system/tensorflow/ -I/home/cgx/git_project/adaptive-system/eigen-eigen implement.cpp ../server/reward.cc ../quantization/util/*.cc ../server/sarsa.cc accuracy.cc ../proto/*.cc -o single_implement_sarsa.bin -L/home/cgx/git_project/adaptive-system/build/ -ltensorflow_cc -L/usr/local/lib -lgrpc++ -lgrpc  -lgrpc++_reflection -L/home/cgx/lib/ -lprotobuf -lpthread -ldl
chmod a+x single_implement_sarsa.bin
