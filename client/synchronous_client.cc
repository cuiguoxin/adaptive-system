#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <string>


#include <grpc++/grpc++.h>

#include "proto/rpc_service.grpc.pb.h"

#include "tensorflow/cc/ops/standard_ops.h"
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
std::map<std::string, std::vector<std::string>> action_to_node_name;

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
  tensorflow::TensorProto const& parameter = tuple.parameter();
  tensorflow::GraphDef const& graph_def = tuple.graph();
  lr = tuple.lr();
  const int interval = tuple.interval();
  const google::protobuf::Map<std::string, std::string> action_to_node_name =
      tuple.action_to_node_name();
  tensorflow::Status tf_status = getSession()->Create(graph_def);
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