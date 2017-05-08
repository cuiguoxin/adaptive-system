g++ -std=c++11 --shared -fPIC -I/home/cgx/git_project/adaptive-system/ -I/home/cgx/include/ -I/home/cgx/git_project/adaptive-system/tensorflow/ -I/home/cgx/git_project/adaptive-system/eigen-eigen algorithms.cc  -o libalgorithms.so
g++ -std=c++11 -I/home/cgx/git_project/adaptive-system/ -I/home/cgx/include/ -I/home/cgx/git_project/adaptive-system/tensorflow/ -I/home/cgx/git_project/adaptive-system/eigen-eigen algorithms_test.cc -o algorithms_test.bin -L/home/cgx/git_project/adaptive-system/build/ -ltensorflow_cc -L/home/cgx/git_project/adaptive-system/quantization/util/ -lalgorithms
export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
cp libalgorithms.so ../../build/
./algorithms_test.bin

