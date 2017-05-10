#include <iostream>
#include <memory>
#include <string>
#include <exception>

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
using grpc::Status;

namespace adaptive_system {
namespace {

std::unique_ptr<SystemControl::Stub> stub;

tensorflow::Session* getSession() {
  static tensorflow::Session* session =
      tensorflow::NewSession(tensorflow::SessionOptions());
  return session;
}

void printError(const Status& status) {
  std::cout << __LINE__ << " line error: error code is " << status.error_code()
            << ", error message is " << status.error_message() << std::endl;
  std::terminate();
}
}
// called in the main
void initStub(std::string const& ip) {
  stub = SystemControl::NewStub(
      grpc::CreateChannel(ip, grpc::InsecureChannelCredentials()));
  std::cout << "init stub ok" << std::endl;
}

void initEverything() {
  Tuple tuple;
  Empty empty;
  ClientContext context;
  Status status = stub->retrieveTuple(&context, empty, &tuple);
  if (!status.ok()) {
    printError(status);
  }
  tensorflow::TensorProto const& parameter = tuple.parameter();
  tensorflow::GraphDef const& graph_def = tuple.graph();
  const float lr = tuple.lr();
  const int interval = tuple.interval();
  const google::protobuf::Map<std::string, std::string> action_to_node_name =
      tuple.action_to_node_name();
}
void closeSession() {
  getSession()->Close();
}
void RunLogic() {}
}

int main(int argc, char* argv[]) {
  std::string ip_port = argv[1];
  adaptive_system::initStub(ip_port);
  adaptive_system::RunLogic();
}