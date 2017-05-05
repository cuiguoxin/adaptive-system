GRPC_PLUGIN_PATH=`which grpc_cpp_plugin`
echo $GRPC_PLUGIN_PATH
protoc -I ../tensorflow/ --grpc_out=../test --plugin=protoc-gen-grpc=$GRPC_PLUGIN_PATH  --proto_path=../proto/ ../proto/helloworld.proto
protoc -I ../tensorflow/ --cpp_out=../test --proto_path=../proto/  ../proto/helloworld.proto
