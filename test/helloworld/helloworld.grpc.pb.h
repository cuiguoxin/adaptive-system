// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: helloworld.proto
#ifndef GRPC_helloworld_2eproto__INCLUDED
#define GRPC_helloworld_2eproto__INCLUDED

#include "helloworld.pb.h"

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/method_handler_impl.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace hw {

// The greeting service definition.
class Greeter final {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    // Sends a greeting
    virtual ::grpc::Status SayHello(::grpc::ClientContext* context, const ::hw::HelloRequest& request, ::hw::HelloReply* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::hw::HelloReply>> AsyncSayHello(::grpc::ClientContext* context, const ::hw::HelloRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::hw::HelloReply>>(AsyncSayHelloRaw(context, request, cq));
    }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::hw::HelloReply>* AsyncSayHelloRaw(::grpc::ClientContext* context, const ::hw::HelloRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status SayHello(::grpc::ClientContext* context, const ::hw::HelloRequest& request, ::hw::HelloReply* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::hw::HelloReply>> AsyncSayHello(::grpc::ClientContext* context, const ::hw::HelloRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::hw::HelloReply>>(AsyncSayHelloRaw(context, request, cq));
    }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader< ::hw::HelloReply>* AsyncSayHelloRaw(::grpc::ClientContext* context, const ::hw::HelloRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::RpcMethod rpcmethod_SayHello_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    // Sends a greeting
    virtual ::grpc::Status SayHello(::grpc::ServerContext* context, const ::hw::HelloRequest* request, ::hw::HelloReply* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_SayHello : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_SayHello() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_SayHello() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHello(::grpc::ServerContext* context, const ::hw::HelloRequest* request, ::hw::HelloReply* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestSayHello(::grpc::ServerContext* context, ::hw::HelloRequest* request, ::grpc::ServerAsyncResponseWriter< ::hw::HelloReply>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_SayHello<Service > AsyncService;
  template <class BaseClass>
  class WithGenericMethod_SayHello : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_SayHello() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_SayHello() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHello(::grpc::ServerContext* context, const ::hw::HelloRequest* request, ::hw::HelloReply* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_SayHello : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_SayHello() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::StreamedUnaryHandler< ::hw::HelloRequest, ::hw::HelloReply>(std::bind(&WithStreamedUnaryMethod_SayHello<BaseClass>::StreamedSayHello, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_SayHello() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status SayHello(::grpc::ServerContext* context, const ::hw::HelloRequest* request, ::hw::HelloReply* response) final override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedSayHello(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::hw::HelloRequest,::hw::HelloReply>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_SayHello<Service > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_SayHello<Service > StreamedService;
};

}  // namespace hw


#endif  // GRPC_helloworld_2eproto__INCLUDED
