#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>

#include "helloworld.grpc.pb.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using hw::HelloRequest;
using hw::HelloReply;
using hw::Greeter;
using tensorflow::Tensor;
using tensorflow::TensorShape;

// Logic and data behind the server's behavior.
class GreeterServiceImpl final : public Greeter::Service {
  Status SayHello(ServerContext *context, const HelloRequest *request,
                  HelloReply *reply) override {
    std::string prefix("Hello ");
    reply->set_message(prefix + request->name());
    const tensorflow::TensorProto &tp = request->tensor_proto();
    Tensor tensor(tensorflow::DT_FLOAT, TensorShape({2, 1}));
    tensor.FromProto(tp);
    auto x_flat = tensor.flat<float>();
    std::cout << x_flat(0) << " " << x_flat(1) << std::endl;
    return Status::OK;
  }
};

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  GreeterServiceImpl service;

  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char **argv) {
  RunServer();

  return 0;
}
