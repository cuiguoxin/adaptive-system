#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>


#include "helloworld.grpc.pb.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using hw::HelloRequest;
using hw::HelloReply;
using hw::Greeter;
using tensorflow::Tensor;
using tensorflow::TensorShape;

class GreeterClient {
 public:
  GreeterClient(std::shared_ptr<Channel> channel)
      : stub_(Greeter::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  std::string SayHello(const std::string& user) {
    // Data we are sending to the server.
    Tensor x(tensorflow::DT_FLOAT, TensorShape({2, 1}));
    auto x_flat = x.flat<float>();
    x_flat.setRandom();
    std::cout << x_flat(0) << " " << x_flat(1) << std::endl;
    tensorflow::TensorProto tp;
    x.AsProtoField(&tp);

    HelloRequest request;
    request.set_name(user);
    request.set_allocated_tensor_proto(&tp);
    

    // Container for the data we expect from the server.
    HelloReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->SayHello(&context, request, &reply);

    request.release_tensor_proto();
    
    // Act upon its status.
    if (status.ok()) {
      return reply.message();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC failed";
    }
  }

 private:
  std::unique_ptr<Greeter::Stub> stub_;
};

int main(int argc, char** argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).
  GreeterClient greeter(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));
  std::string user("world");
  std::string reply = greeter.SayHello(user);
  std::cout << "Greeter received: " << reply << std::endl;

  return 0;
}
