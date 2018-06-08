GRPC_PLUGIN_PATH=`which grpc_cpp_plugin` 
OUT_PATH=~/git_project/adaptive-system/proto
PYTHON_OUT_PATH=~/git_project/adaptive-system/input/cifar10
echo $GRPC_PLUGIN_PATH 
protoc -I ~/git_project/adaptive-system/tensorflow/ --grpc_out=$OUT_PATH --plugin=protoc-gen-grpc=$GRPC_PLUGIN_PATH  --proto_path=$OUT_PATH $OUT_PATH/rpc_service.proto 
protoc -I ~/git_project/adaptive-system/tensorflow/ --cpp_out=$OUT_PATH --python_out=$PYTHON_OUT_PATH --proto_path=$OUT_PATH  $OUT_PATH/rpc_service.proto 
