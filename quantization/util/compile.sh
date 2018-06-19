g++ -std=c++11 --shared -fPIC -I/home/cgx/git_project/adaptive-system/ -I/home/cgx/include/ -I/home/cgx/git_project/adaptive-system/tensorflow/ -I/home/cgx/git_project/adaptive-system/eigen-eigen algorithms.cc extract_feature.cc any_level.cc -o libalgorithms.so
export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/
cp libalgorithms.so ../../build/

