export LD_LIBRARY_PATH=/home/cgx/git_project/adaptive-system/build/:/home/cgx/grpc/libs/opt/:/home/cgx/lib
chmod a+x client.bin
./client.bin 10.61.1.120:50051 /home/cgx/git_project/adaptive-system/resources/cifar-10-batches-bin/data_batch.bin /home/cgx/git_project/adaptive-system/input/cifar10/preprocess.pb
