GRPC_PLUGIN_PATH=`which grpc_cpp_plugin`
echo $GRPC_PLUGIN_PATH
protoc -I /home/cgx/git_project/adaptive-system/tensorflow/ --grpc_out=/home/cgx/git_project/adaptive-system/test/helloworld --plugin=protoc-gen-grpc=$GRPC_PLUGIN_PATH  --proto_path=/home/cgx/git_project/adaptive-system/proto/ /home/cgx/git_project/adaptive-system/proto/helloworld.proto
protoc -I /home/cgx/git_project/adaptive-system/tensorflow/ --cpp_out=/home/cgx/git_project/adaptive-system/test/helloworld --proto_path=/home/cgx/git_project/adaptive-system/proto/  /home/cgx/git_project/adaptive-system/proto/helloworld.proto
