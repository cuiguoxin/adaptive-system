#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>

#include "proto/rpc_service.grpc.pb.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

namespace adaptive_system {
class RPCServiceImpl final : public SystemControl::Service {
  Status retrieveTuple(ServerContext* context,
                       const HelloRequest* request,
                       HelloReply* reply) override {
    return Status::OK;
  }

  Status sendGradient(ServerContext* context,
                      GradientAndLoss* request,
                      Gradient* response) override {
    return Status::OK;
  }

  Status sendState(ServerContext* context,
                   PartialStateAndLoss* request,
                   QuantizationLevel* response) {
    return Status::OK;
  }
};
}

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  RPCServiceImpl service;

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

int main(int argc, char** argv) {
  RunServer();

  return 0;
}
