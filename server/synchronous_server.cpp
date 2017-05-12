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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#include "quantization/util/algorithms.h"

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
        _tuple_local_path(tuple_local_path) {
    session = tensorflow::NewSession(tensorflow::SessionOptions());
    std::fstream input(_tuple_local_path, std::ios::in | std::ios::binary);
    if (!input) {
      std::cout << _tuple_local_path
                << ": File not found.  Creating a new file." << std::endl;
    } else if (!tuple.ParseFromIstream(&input)) {
      std::cerr << "Failed to parse tuple." << std::endl;
      std::terminate();
    }
  }
  Status retrieveTuple(ServerContext* context, const Empty* request,
                       Tuple* reply) override {
    *reply = tuple;
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
    _bool_gradient = false;
    _vector_loss.push_back(loss);
    _vector_map_gradient.push_back(map_gradient);
    if (_vector_map_gradient.size() == _number_of_workers) {
      std::map<std::string, tensorflow::Tensor> map_gradient_another;
      aggregate_gradients(_vector_map_gradient, map_gradient_another);
      do_quantization(map_gradient_another, &_store_named_gradient);
      apply_quantized_gradient_to_model(_store_named_gradient);
      _vector_map_gradient.clear();
      _bool_gradient = true;
      _condition_variable_gradient.notify_all();
    } else {
      _condition_variable_gradient.wait(lk, [this] { return _bool_gradient; });
    }
    lk.unlock();
    *response = _store_named_gradient;
    return Status::OK;
  }

  Status sendState(ServerContext* context, const PartialStateAndLoss* request,
                   QuantizationLevel* response) {
    std::unique_lock<std::mutex> lk(_mutex_state);
    _vector_partial_state_and_loss.push_back(*request);
    if (_vector_partial_state_and_loss.size() == _number_of_workers) {
      adjust_rl_model(_vector_partial_state_and_loss);
    } else {
      _condition_variable_state.wait(lk, [this] { return _bool_state; });
    }
    lk.unlock();
    response->set_level(_grad_quant_level);
    return Status::OK;
  }

  // private member functions
 private:
  // put named quantized gradients into a dequantized tensor map
  void convert_named_gradient_to_map_gradient(
      const NamedGradients& named_gradients,
      std::map<std::string, tensorflow::Tensor>& map_gradient) {
    auto map_quantized_gradient =
        named_gradients.name_to_gradient();  // const reference
    std::for_each(
        map_quantized_gradient.cbegin(), map_quantized_gradient.cend(),
        [&map_gradient](
            ::google::protobuf::MapPair<std::string, Gradient> const& pair) {
          Gradient& gradient = const_cast<Gradient&>(pair.second);
          std::string const& variable_name = pair.first;
          tensorflow::Tensor temp_tensor;
          dequantize(
              cast_grad_quant_level_to_quantization_type(gradient.level()),
              gradient, temp_tensor);
          map_gradient.insert(std::pair<std::string, tensorflow::Tensor>(
              variable_name, temp_tensor));
        });
  }

  void aggregate_gradients(
      std::vector<std::map<std::string, tensorflow::Tensor>> const&
          vector_map_gradient,
      std::map<std::string, tensorflow::Tensor>& map_gradient) {
    std::for_each(
        vector_map_gradient.cbegin(), vector_map_gradient.cend(),
        [&map_gradient](
            std::map<std::string, tensorflow::Tensor> const& current_map) {
          std::for_each(
              current_map.cbegin(), current_map.cend(),
              [&map_gradient](
                  std::pair<std::string, tensorflow::Tensor> const& pair) {
                std::string const& variable_name = pair.first;
                tensorflow::Tensor const& tensor_to_be_aggregate = pair.second;
                const float* tensor_to_be_aggregate_ptr =
                    tensor_to_be_aggregate.flat<float>().data();
                auto iter = map_gradient.find(variable_name);
                if (iter == map_gradient.end()) {
                  tensorflow::Tensor new_tensor(tensorflow::DataType::DT_FLOAT,
                                                tensor_to_be_aggregate.shape());
                  float* new_tensor_ptr = new_tensor.flat<float>().data();
                  size_t num_new_tensor = new_tensor.NumElements();
                  std::copy(tensor_to_be_aggregate_ptr,
                            tensor_to_be_aggregate_ptr + num_new_tensor,
                            new_tensor_ptr);
                  map_gradient.insert(
                      std::make_pair(variable_name, new_tensor));
                } else {
                  tensorflow::Tensor& tensor_sum = iter->second;
                  float* tensor_sum_ptr = tensor_sum.flat<float>().data();
                  size_t num_new_tensor = tensor_sum.NumElements();
                  for (size_t i = 0; i < num_new_tensor; i++) {
                    tensor_sum_ptr[i] += tensor_to_be_aggregate_ptr[i];
                  }
                }
              });
        });
  }
  // map_gradient---->>>named_gradients
  void do_quantization(std::map<std::string, tensorflow::Tensor>& map_gradient,
                       NamedGradients* named_gradients) {
    std::for_each(map_gradient.begin(), map_gradient.end(),
                  [named_gradients, this](
                      std::pair<std::string const, tensorflow::Tensor>& pair) {
                    std::string const& variable_name = pair.first;
                    tensorflow::Tensor& raw_tensor = pair.second;
                    float max = 0, min = 0;
                    get_max_and_min_value(raw_tensor, max, min);
                    Gradient grad;
                    quantize(cast_grad_quant_level_to_quantization_type(
                                 _grad_quant_level),
                             raw_tensor, max, min, grad);
                    named_gradients->mutable_name_to_gradient()->insert(
                        google::protobuf::MapPair<::std::string, Gradient>(
                            variable_name, grad));
                  });
  }

  void apply_quantized_gradient_to_model(NamedGradients& named_gradients) {
    google::protobuf::Map<std::string, Gradient>& map_gradient =
        *named_gradients.mutable_name_to_gradient();
    google::protobuf::Map<std::string, Names> const& map_names =
        tuple.map_names();
    std::vector<std::pair<std::string, tensorflow::Tensor>> feeds;
    std::vector<std::string> actions_to_do;
    std::for_each(
        map_gradient.begin(), map_gradient.end(),
        [&feeds, &actions_to_do, &map_names,
         this](google::protobuf::MapPair<std::string, Gradient>& pair) {
          std::string const& variable_name = pair.first;
          Gradient& grad = pair.second;
          auto iter_map_names = map_names.find(variable_name);
          if (iter_map_names == map_names.end()) {
            std::cout << "this is impossible Line " << __LINE__ << std::endl;
            std::terminate();
          } else {
            Names& names = iter->second;
            std::string assign_add_name = names.assign_add_name();
            tensorflow::TensorShape shape(
                tuple.map_parameters.find(variable_name)
                    ->second.tensor_shape());
            tensorflow::Tensor feed(tensorflow::DataType::DT_FLOAT, shape);
            dequantize(
                cast_grad_quant_level_to_quantization_type(_grad_quant_level),
                grad, feed);
            feeds.push_back(std::pair<std::string, tensorflow::Tensor>(
                names.placeholder_assign_add_name(), feed));
            actions_to_do.push_back(assign_add_name);
          }
        });
    std::vector<tensorflow::Tensor> output;
    session->Run(feeds, {}, actions_to_do, &output);
  }

  void adjust_rl_model(
      std::vector<PartialStateAndLoss> const& vector_partial_state_and_loss) {}
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
  bool _bool_gradient;
  bool _bool_state;
  NamedGradients _store_named_gradient;

  tensorflow::Session* session;
  Tuple tuple;
};
}

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  adaptive_system::RPCServiceImpl service(
      3, 0.1f, 5000, 3, adaptive_system::GRAD_QUANT_LEVEL::EIGHT, "");

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
