export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
g++ -std=c++11 -I/home/cgx/git_project/adaptive-system/ -I/home/cgx/include/ -I/home/cgx/git_project/adaptive-system/tensorflow/ -I/home/cgx/git_project/adaptive-system/eigen-eigen implement.cpp ../proto/*.cc accuracy.cc -o single_implement_qsgd_baseline_combine.bin -L/home/cgx/git_project/adaptive-system/build/ -ltensorflow_cc -lalgorithms -L/usr/local/lib -lgrpc++ -lgrpc  -lgrpc++_reflection -L/home/cgx/lib/ -lprotobuf -lpthread -ldl
chmod a+x single_implement_qsgd_baseline_combine.bin
