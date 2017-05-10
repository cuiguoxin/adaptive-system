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

using grpc::Channel;
using grpc::ClientContext;

namespace adaptive_system {
namespace {

std::unique_ptr<SystemControl::Stub> stub;
float lr = 0.0;
int interval = 0;
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
  std::vector<tensorflow::Tensor> outputs;
  tf_status = get_session()->Run(feeds, {}, assign_names, &outputs);
  if (!tf_status.ok()) {
    print_error(tf_status);
  }
}

void close_session() { get_session()->Close(); }

void run_logic() {}
}

int main(int argc, char* argv[]) {
  std::string ip_port = argv[1];
  adaptive_system::init_stub(ip_port);
  adaptive_system::run_logic();
}
