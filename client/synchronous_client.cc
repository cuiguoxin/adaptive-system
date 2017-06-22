#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <thread>

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

#include "input/word2vec/input.h"
#include "quantization/util/algorithms.h"
#include "quantization/util/extract_feature.h"
#include "quantization/util/helper.h"

using grpc::Channel;
using grpc::ClientContext;

namespace adaptive_system {
	namespace {

		std::unique_ptr<SystemControl::Stub> stub;
		float lr = 0.0;
		int interval = 0;
		int total_iter = 1000;
		size_t batch_size = 0;
		int grad_quant_level_order = 0;
		std::string label_placeholder_name, batch_placeholder_name;
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
	}
	// called in the main
	void init_stub(std::string const& ip) {
		grpc::ChannelArguments channel_args;
		channel_args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH,
			std::numeric_limits<int>::max());
		stub = SystemControl::NewStub(grpc::CreateCustomChannel(
			ip, grpc::InsecureChannelCredentials(), channel_args));
		std::cout << "init stub ok" << std::endl;
	}

	void init_everything() {
		Tuple tuple;
		Empty empty;
		ClientContext context;
		grpc::Status grpc_status = stub->retrieveTuple(&context, empty, &tuple);
		if (!grpc_status.ok()) {
			PRINT_ERROR_MESSAGE(grpc_status.error_message());
			std::terminate();
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
		total_iter = tuple.total_iter();
		batch_size = tuple.batch_size();
		std::string init_name = tuple.init_name();
		batch_placeholder_name = tuple.batch_placeholder_name();
		label_placeholder_name = tuple.label_placeholder_name();
		tensorflow::Status tf_status = get_session()->Create(graph_def);
		get_session()->Run({}, {}, { init_name }, nullptr);
		if (!tf_status.ok()) {
			PRINT_ERROR_MESSAGE(tf_status.error_message());
			std::terminate();
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
			PRINT_ERROR_MESSAGE(tf_status.error_message());
			std::terminate();
		}
		*get_tuple() = tuple;
	}
	// return loss and set gradient to the first parameter
	float compute_gradient_and_loss(
		std::vector<std::pair<std::string, tensorflow::Tensor>> feeds,
		std::map<std::string, tensorflow::Tensor>& gradients,
		std::map<std::string, tensorflow::Tensor>& indices) {

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
			fetch.push_back(names.gradient_index_name());
			variable_names_in_order.push_back(variable_name);
		});
		tensorflow::Status tf_status = get_session()->Run(feeds, fetch, {}, &outputs);
		if (!tf_status.ok()) {
			PRINT_ERROR_MESSAGE(tf_status.error_message());
			std::terminate();
		}
		tensorflow::Tensor& loss_tensor = outputs[0];
		float* loss_ptr = loss_tensor.flat<float>().data();
		float loss_ret = loss_ptr[0];
		outputs.erase(outputs.begin());

		size_t size = outputs.size();
		for (size_t i = 0; i < size; i = i + 2) {
			gradients.insert(std::pair<std::string, tensorflow::Tensor>(
				variable_names_in_order[i / 2], outputs[i]));
			indices.insert(std::pair<std::string, tensorflow::Tensor>(
				variable_names_in_order[i / 2], outputs[i + 1]));
		}
		if (gradients.size() != indices.size()) {
			PRINT_ERROR_MESSAGE("value and index's sizes are not the same");
			std::terminate();
		}
		return loss_ret;
	}
	// do not need session, currently not use loss information
	PartialState collect_partial_state(
		std::map<std::string, tensorflow::Tensor> const& gradients,
		const float loss) {
		PartialState partial_state_ret;
		static const std::string variable_name_to_collect = "Variable:0";
		auto iter = gradients.find(variable_name_to_collect);
		if (iter == gradients.end()) {
			PRINT_ERROR;
			std::terminate();
		}
		else {
			tensorflow::Tensor const & tensor_to_be_collected = iter->second;
			tensorflow::Tensor feature_tensor = get_feature(tensor_to_be_collected, loss);
			tensorflow::TensorProto feature_tensor_proto;
			feature_tensor.AsProtoField(&feature_tensor_proto);
			*partial_state_ret.mutable_tensor() = feature_tensor_proto;
		}
		return partial_state_ret;
	}

	//void show_quantization_infor(
	//	std::map<std::string, tensorflow::Tensor>& map_gradients,
	//	NamedGradients& named_gradients_send) {
	//	std::map<std::string, tensorflow::Tensor> map_gradients_other;
	//	dequantize_gradient(named_gradients_send, map_gradients_other);
	//	std::for_each(
	//		map_gradients.begin(), map_gradients.end(),
	//		[&map_gradients_other](std::pair<std::string, tensorflow::Tensor> pair) {
	//		std::string variable_name = pair.first;
	//		tensorflow::Tensor& tensor = pair.second;
	//		float* tensor_ptr = tensor.flat<float>().data();
	//		size_t size = tensor.NumElements();
	//		auto iter = map_gradients_other.find(variable_name);
	//		tensorflow::Tensor& tensor_other = iter->second;
	//		float* tensor_other_ptr = tensor_other.flat<float>().data();
	//		std::cout << variable_name << " : ";
	//		for (int i = 0; i < 10; i++) {
	//			std::cout << "(" << tensor_other_ptr[i] << "-" << tensor_ptr[i] << "="
	//				<< tensor_other_ptr[i] - tensor_ptr[i] << "), ";
	//		}
	//		std::cout << std::endl;
	//	});
	//}

	/*void now_sleep(GRAD_QUANT_LEVEL level) {
		switch (level) {
		case GRAD_QUANT_LEVEL::ONE:
			std::this_thread::sleep_for(std::chrono::duration<float>(0.5f));
			break;
		case GRAD_QUANT_LEVEL::TWO:
			std::this_thread::sleep_for(std::chrono::duration<float>(1.0f));
			break;
		case GRAD_QUANT_LEVEL::FOUR:
			std::this_thread::sleep_for(std::chrono::duration<float>(1.5f));
			break;
		case GRAD_QUANT_LEVEL::EIGHT:
			std::this_thread::sleep_for(std::chrono::duration<float>(2.5f));
			break;
		case GRAD_QUANT_LEVEL::SIXTEEN:
			std::this_thread::sleep_for(std::chrono::duration<float>(4.0f));
			break;
		case GRAD_QUANT_LEVEL::NONE:
			std::this_thread::sleep_for(std::chrono::duration<float>(6.0f));
			break;
		}
	}*/


	void do_training(std::string const & raw_data_path) {
		word2vec::init(raw_data_path, get_tuple()->word_to_index());
		for (int i = 0; i < total_iter; i++) {
			PRINT_INFO;
			std::map<std::string, tensorflow::Tensor> map_gradients;
			std::map<std::string, tensorflow::Tensor> map_indices;
			std::pair<tensorflow::Tensor, tensorflow::Tensor> feeds =
				word2vec::get_next_batch(batch_size);
			PRINT_INFO;
			float loss = compute_gradient_and_loss(
			{ {batch_placeholder_name, feeds.first},
			 {label_placeholder_name, feeds.second} },
				map_gradients, map_indices);  // compute gradient and loss now
			PRINT_INFO;
			Loss loss_to_send;
			loss_to_send.set_loss(loss);
			Empty empty;
			ClientContext loss_context;
			PRINT_INFO;
			stub->sendLoss(&loss_context, loss_to_send, &empty);
			PRINT_INFO;
			if (i % interval == 0) {
				PartialState partial_state = collect_partial_state(map_gradients, loss);
				ClientContext state_context;
				QuantizationLevel quantization_level;
				PRINT_INFO;
				grpc::Status grpc_status =
					stub->sendState(&state_context, partial_state, &quantization_level);
				PRINT_INFO;
				if (!grpc_status.ok()) {
					PRINT_ERROR_MESSAGE(grpc_status.error_message());
					std::terminate();
				}
				grad_quant_level_order = quantization_level.level_order();
			}
			//fake
			//now_sleep(grad_quant_level);
			NamedGradients named_gradients_send, named_gradients_receive;
			PRINT_INFO;
			quantize_gradients(
				map_gradients, &named_gradients_send,
				get_tuple()->order_to_level().find(grad_quant_level_order)->second);
			PRINT_INFO;
			add_indices_to_named_gradients(map_indices, named_gradients_send);
			ClientContext gradient_context;
			PRINT_INFO;
			grpc::Status grpc_status = stub->sendGradient(
				&gradient_context, named_gradients_send, &named_gradients_receive);
			//show_quantization_infor(map_gradients, named_gradients_receive);
			PRINT_INFO;
			if (!grpc_status.ok()) {
				PRINT_ERROR_MESSAGE(grpc_status.error_message());
				std::terminate();
			}
			PRINT_INFO;
			// add the gradients to variables
			apply_quantized_gradient_to_model(named_gradients_receive,
				get_session(), *get_tuple());
			PRINT_INFO;
		}
	}

	void close_session() { get_session()->Close(); }

	void run_logic(std::string const & raw_data_path) {
		init_everything();
		do_training(raw_data_path);
		close_session();
	}
}

int main(int argc, char* argv[]) {
	std::string ip_port = argv[1];
	std::string raw_data_path = argv[2];
	adaptive_system::init_stub(ip_port);
	adaptive_system::run_logic(raw_data_path);
}
