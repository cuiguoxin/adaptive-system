// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: helloworld.proto

#include "helloworld.pb.h"
#include "helloworld.grpc.pb.h"

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/channel_interface.h>
#include <grpc++/impl/codegen/client_unary_call.h>
#include <grpc++/impl/codegen/method_handler_impl.h>
#include <grpc++/impl/codegen/rpc_service_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/sync_stream.h>
namespace tensorflow {

static const char* Greeter_method_names[] = {
  "/tensorflow.Greeter/SayHello",
};

std::unique_ptr< Greeter::Stub> Greeter::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  std::unique_ptr< Greeter::Stub> stub(new Greeter::Stub(channel));
  return stub;
}

Greeter::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel)
  : channel_(channel), rpcmethod_SayHello_(Greeter_method_names[0], ::grpc::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status Greeter::Stub::SayHello(::grpc::ClientContext* context, const ::tensorflow::HelloRequest& request, ::tensorflow::HelloReply* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_SayHello_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::tensorflow::HelloReply>* Greeter::Stub::AsyncSayHelloRaw(::grpc::ClientContext* context, const ::tensorflow::HelloRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::tensorflow::HelloReply>(channel_.get(), cq, rpcmethod_SayHello_, context, request);
}

Greeter::Service::Service() {
  AddMethod(new ::grpc::RpcServiceMethod(
      Greeter_method_names[0],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< Greeter::Service, ::tensorflow::HelloRequest, ::tensorflow::HelloReply>(
          std::mem_fn(&Greeter::Service::SayHello), this)));
}

Greeter::Service::~Service() {
}

::grpc::Status Greeter::Service::SayHello(::grpc::ServerContext* context, const ::tensorflow::HelloRequest* request, ::tensorflow::HelloReply* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace tensorflow

