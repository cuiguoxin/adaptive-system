#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
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

#include "input/cifar10/input.h"
#include "quantization/util/algorithms.h"

using grpc::Channel;
using grpc::ClientContext;

namespace adaptive_system {
namespace {

std::unique_ptr<SystemControl::Stub> stub;
float lr = 0.0;
int interval = 0;
int total_iter = 0;
GRAD_QUANT_LEVEL grad_quant_level = GRAD_QUANT_LEVEL::NONE;
std::string label_placeholder_name, image_placeholder_name;
Tuple* get_tuple() {
  static Tuple tuple;
  return &tuple;
}

std::map<std::string, Names>* get_map_names() {
  static std::map<std::string, Names> map_names;
  return &map_names;
}

tensorflow::Session* get_session() {
  static tensorflow::Session* session =
      tensorflow::NewSession(tensorflow::SessionOptions());
  return session;
}

void print_error(const grpc::Status& status) {
  std::cout << __LINE__ << " line error: error code is " << status.error_code()
            << ", error message is " << status.error_message() << std::endl;
  std::terminate();
}
void print_error(const tensorflow::Status& status) {
  std::cout << __LINE__ << " line error: error code is " << status.code()
            << ", error message is " << status.error_message() << std::endl;
  std::terminate();
}
}
// called in the main
void init_stub(std::string const& ip) {
  stub = SystemControl::NewStub(
      grpc::CreateChannel(ip, grpc::InsecureChannelCredentials()));
  std::cout << "init stub ok" << std::endl;
}

void init_everything() {
  Tuple tuple;
  Empty empty;
  ClientContext context;
  grpc::Status grpc_status = stub->retrieveTuple(&context, empty, &tuple);
  if (!grpc_status.ok()) {
    print_error(grpc_status);
  }
  // init map_names
  google::protobuf::Map<std::string, Names> const& map_names =
      tuple.map_names();
  std::for_each(map_names.cbegin(), map_names.cend(),
                [](google::protobuf::MapPair<std::string, Names> const& p) {
                  get_map_names()->insert(p);
                });

  tensorflow::GraphDef const& graph_def = tuple.graph();
  lr = tuple.lr();
  interval = tuple.interval();
  image_placeholder_name = tuple.image_placeholder_name();
  label_placeholder_name = tuple.label_placeholder_name();
  tensorflow::Status tf_status = get_session()->Create(graph_def);
  if (!tf_status.ok()) {
    print_error(tf_status);
  }
  // init all the variables
  google::protobuf::Map<std::string, tensorflow::TensorProto> const&
      map_parameters = tuple.map_parameters();
  std::vector<std::string> assign_names;
  std::vector<std::pair<std::string, tensorflow::Tensor>> feeds;
  std::for_each(
      map_parameters.cbegin(), map_parameters.cend(),
      [&assign_names, &feeds](
          google::protobuf::MapPair<std::string, tensorflow::TensorProto> const&
              pair) {
        tensorflow::Tensor tensor;
        bool is_success = tensor.FromProto(pair.second);
        if (!is_success) {
          std::terminate();
        }
        auto iter = get_map_names()->find(pair.first);
        std::string assign_name = (iter->second).assign_name();
        std::string placeholder_name = (iter->second).placeholder_assign_name();
        assign_names.push_back(assign_name);
        feeds.push_back(std::make_pair(placeholder_name, tensor));

      });
  tf_status = get_session()->Run(feeds, {}, assign_names, nullptr);
  if (!tf_status.ok()) {
    print_error(tf_status);
  }
  *get_tuple() = tuple;
}
// return loss and set gradient to the first parameter
float compute_gradient_and_loss(
    std::vector<std::pair<std::string, tensorflow::Tensor>> feeds,
    std::map<std::string, tensorflow::Tensor>& gradients) {
  std::vector<std::string> fetch;
  std::string loss_name = get_tuple()->loss_name();
  fetch.push_back(loss_name);
  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::string> variable_names_in_order;
  google::protobuf::Map<std::string, Names> const& map_names =
      get_tuple()->map_names();
  std::for_each(map_names.begin(), map_names.end(),
                [&fetch, &variable_names_in_order](
                    google::protobuf::MapPair<std::string, Names> const& pair) {
                  Names const& names = pair.second;
                  std::string const& variable_name = pair.first;
                  fetch.push_back(names.gradient_name());
                  variable_names_in_order.push_back(variable_name);
                });
  get_session()->Run(feeds, fetch, {}, &outputs);
  tensorflow::Tensor& loss_tensor = outputs[0];
  float* loss_ptr = loss_tensor.flat<float>().data();
  outputs.erase(outputs.begin());
  if (outputs.size() != variable_names_in_order.size()) {
    std::cout << "impossible in " << __LINE__ << std::endl;
    std::terminate();
  }
  size_t size = outputs.size();
  for (size_t i = 0; i < size; i++) {
    gradients.insert(std::pair<std::string, tensorflow::Tensor>(
        variable_names_in_order[i], outputs[i]));
  }
}
// do not need session
PartialState collect_partial_state(
    std::map<std::string, tensorflow::Tensor> const& gradients,
    const float loss) {}

void do_training(const std::string& binary_file_path,
                 const std::string& graph_path) {
  cifar10::turn_raw_tensors_to_standard_version(binary_file_path, graph_path);
  for (int i = 0; i < total_iter; i++) {
    std::map<std::string, tensorflow::Tensor> map_gradients;
    std::pair<tensorflow::Tensor, tensorflow::Tensor> feeds =
        cifar10::get_next_batch();
    float loss = compute_gradient_and_loss(
        {{image_placeholder_name, feeds.first},
         {label_placeholder_name, feeds.second}},
        map_gradients);  // compute gradient and loss now
    Loss loss_to_send;
    loss_to_send.set_loss(loss);
    Empty empty;
    ClientContext loss_context;
    stub->sendLoss(&loss_context, loss_to_send, &empty);
    if (i % interval == 0) {
      PartialState partial_state = collect_partial_state(map_gradients, loss);
      ClientContext state_context;
      QuantizationLevel quantization_level;
      grpc::Status grpc_status =
          stub->sendState(&state_context, partial_state, &quantization_level);
      if (!grpc_status.ok()) {
        print_error(grpc_status);
      }
      grad_quant_level = quantization_level.level();
    }
    NamedGradients named_gradients_send, named_gradients_receive;
    quantize_gradient(
        map_gradients, &named_gradients_send,
        cast_grad_quant_level_to_quantization_type(grad_quant_level));
    ClientContext gradient_context;
    grpc::Status grpc_status = stub->sendGradient(
        &gradient_context, named_gradients_send, &named_gradients_receive);
    if (!grpc_status.ok()) {
      print_error(grpc_status);
    }
    // add the gradients to variables
    apply_quantized_gradient_to_model(named_gradients_receive, get_session(),
                                      *get_tuple());
  }
}

void close_session() { get_session()->Close(); }

void run_logic() {
  init_everything();
  do_training();
  close_session();
}
}

int main(int argc, char* argv[]) {
  std::string ip_port = argv[1];
  adaptive_system::init_stub(ip_port);
  adaptive_system::run_logic();
}
