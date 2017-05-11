#include <algorithm>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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
 public:
  RPCServiceImpl(int interval, float lr, int total_iter, int number_of_workers,
                 GRAD_QUANT_LEVEL grad_quant_level,
                 std::string const& tuple_local_path)
      : SystemControl::Service(),
        _interval(interval),
        _lr(lr),
        _total_iter(total_iter),
        _number_of_workers(number_of_workers),
        _grad_quant_level(grad_quant_level),
        _tuple_local_path(tuple_local_path) {}
  Status retrieveTuple(ServerContext* context, const Empty* request,
                       Tuple* reply) override {
    fstream input(_tuple_local_path, ios::in | ios::binary);
    if (!input) {
      std::cout << _tuple_local_path
                << ": File not found.  Creating a new file." << std::endl;
    } else if (!reply->ParseFromIstream(&input)) {
      std::cerr << "Failed to parse tuple." << std::endl;
      std::terminate();
    }
    return Status::OK;
  }

  Status sendGradient(ServerContext* context,
                      const NamedGradientsAndLoss* request,
                      NamedGradients* response) override {
    const NamedGradients& named_gradients = request->named_gradients();
    const float loss = request->loss();
    std::map<std::string, tensorflow::Tensor> map_gradient;
    convert_named_gradient_to_map_gradient(named_gradients, map_gradient);
    std::unique_lock<std::mutex> lk(_mutex_gradient);
    _vector_map_gradient.push_back(map_gradient);
    _condition_variable_gradient.wait(lk, [this] {
      return _vector_map_gradient.size() == _number_of_workers;
    });
    return Status::OK;
  }

  Status sendState(ServerContext* context, const PartialStateAndLoss* request,
                   QuantizationLevel* response) {
    return Status::OK;
  }

  // private member functions
 private:
  void convert_named_gradient_to_map_gradient(
      const NamedGradients& named_gradients,
      std::map<std::string, tensorflow::Tensor>& map_gradient) {}

  // private data member
 private:
  const int _interval;
  const float _lr;
  const int _total_iter;
  const int _number_of_workers;
  GRAD_QUANT_LEVEL _grad_quant_level = GRAD_QUANT_LEVEL::NONE;

  std::mutex _mutex_gradient;
  std::mutex _mutex_state;
  std::condition_variable _condition_variable_gradient;
  std::condition_variable _condition_variable_state;
  std::vector<std::map<std::string, tensorflow::Tensor>> _vector_map_gradient;
  std::vector<PartialStateAndLoss> _vector_partial_state_and_loss;
  float _last_loss;
  std::vector<float> _vector_loss;
  std::string _tuple_local_path;
};
}

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  adaptive_system::RPCServiceImpl service;

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
