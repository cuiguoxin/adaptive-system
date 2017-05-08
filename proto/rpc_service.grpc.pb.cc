// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: rpc_service.proto

#include "rpc_service.pb.h"
#include "rpc_service.grpc.pb.h"

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/channel_interface.h>
#include <grpc++/impl/codegen/client_unary_call.h>
#include <grpc++/impl/codegen/method_handler_impl.h>
#include <grpc++/impl/codegen/rpc_service_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/sync_stream.h>
namespace adaptive_system {

static const char* SystemControl_method_names[] = {
  "/adaptive_system.SystemControl/retrieveTuple",
  "/adaptive_system.SystemControl/sendGradient",
  "/adaptive_system.SystemControl/sendState",
};

std::unique_ptr< SystemControl::Stub> SystemControl::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  std::unique_ptr< SystemControl::Stub> stub(new SystemControl::Stub(channel));
  return stub;
}

SystemControl::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel)
  : channel_(channel), rpcmethod_retrieveTuple_(SystemControl_method_names[0], ::grpc::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_sendGradient_(SystemControl_method_names[1], ::grpc::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_sendState_(SystemControl_method_names[2], ::grpc::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status SystemControl::Stub::retrieveTuple(::grpc::ClientContext* context, const ::adaptive_system::Empty& request, ::adaptive_system::TuplePaLrItv* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_retrieveTuple_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::adaptive_system::TuplePaLrItv>* SystemControl::Stub::AsyncretrieveTupleRaw(::grpc::ClientContext* context, const ::adaptive_system::Empty& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::adaptive_system::TuplePaLrItv>(channel_.get(), cq, rpcmethod_retrieveTuple_, context, request);
}

::grpc::Status SystemControl::Stub::sendGradient(::grpc::ClientContext* context, const ::adaptive_system::GradientAndLoss& request, ::adaptive_system::Gradient* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_sendGradient_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::adaptive_system::Gradient>* SystemControl::Stub::AsyncsendGradientRaw(::grpc::ClientContext* context, const ::adaptive_system::GradientAndLoss& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::adaptive_system::Gradient>(channel_.get(), cq, rpcmethod_sendGradient_, context, request);
}

::grpc::Status SystemControl::Stub::sendState(::grpc::ClientContext* context, const ::adaptive_system::PartialStateAndLoss& request, ::adaptive_system::QuantizationLevel* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_sendState_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::adaptive_system::QuantizationLevel>* SystemControl::Stub::AsyncsendStateRaw(::grpc::ClientContext* context, const ::adaptive_system::PartialStateAndLoss& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::adaptive_system::QuantizationLevel>(channel_.get(), cq, rpcmethod_sendState_, context, request);
}

SystemControl::Service::Service() {
  AddMethod(new ::grpc::RpcServiceMethod(
      SystemControl_method_names[0],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< SystemControl::Service, ::adaptive_system::Empty, ::adaptive_system::TuplePaLrItv>(
          std::mem_fn(&SystemControl::Service::retrieveTuple), this)));
  AddMethod(new ::grpc::RpcServiceMethod(
      SystemControl_method_names[1],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< SystemControl::Service, ::adaptive_system::GradientAndLoss, ::adaptive_system::Gradient>(
          std::mem_fn(&SystemControl::Service::sendGradient), this)));
  AddMethod(new ::grpc::RpcServiceMethod(
      SystemControl_method_names[2],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< SystemControl::Service, ::adaptive_system::PartialStateAndLoss, ::adaptive_system::QuantizationLevel>(
          std::mem_fn(&SystemControl::Service::sendState), this)));
}

SystemControl::Service::~Service() {
}

::grpc::Status SystemControl::Service::retrieveTuple(::grpc::ServerContext* context, const ::adaptive_system::Empty* request, ::adaptive_system::TuplePaLrItv* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status SystemControl::Service::sendGradient(::grpc::ServerContext* context, const ::adaptive_system::GradientAndLoss* request, ::adaptive_system::Gradient* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status SystemControl::Service::sendState(::grpc::ServerContext* context, const ::adaptive_system::PartialStateAndLoss* request, ::adaptive_system::QuantizationLevel* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace adaptive_system

