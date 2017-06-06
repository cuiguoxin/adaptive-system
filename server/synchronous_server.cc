#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
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
#include "quantization/util/helper.h"
#include "quantization/util/extract_feature.h"

#include "server/sarsa.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

namespace adaptive_system {
	class RPCServiceImpl final : public SystemControl::Service {
	private:
		void print_state_to_file(tensorflow::Tensor const & state) {
			size_t feature_number = state.NumElements();
			const float * state_ptr = state.flat<float>().data();
			for (size_t i = 0; i < feature_number; i++) {
				_file_state_stream << std::to_string(state_ptr[i]) << " ";
			}
			_file_state_stream << "\n";
			_file_state_stream.flush();
		}
	public:
		RPCServiceImpl(int interval, float lr, int total_iter, int number_of_workers,
			GRAD_QUANT_LEVEL grad_quant_level,
			std::string const& tuple_local_path,
			std::string const & sarsa_path, float r, float eps_greedy)
			: SystemControl::Service(),
			_interval(interval),
			_lr(lr),
			_total_iter(total_iter),
			_number_of_workers(number_of_workers),
			_grad_quant_level(grad_quant_level),
			_tuple_local_path(tuple_local_path),
			_sarsa(sarsa_path, r, eps_greedy)
		{
			_session = tensorflow::NewSession(tensorflow::SessionOptions());
			std::fstream input(_tuple_local_path, std::ios::in | std::ios::binary);
			if (!input) {
				std::cout << _tuple_local_path
					<< ": File not found.  Creating a new file." << std::endl;
			}
			else if (!_tuple.ParseFromIstream(&input)) {
				std::cerr << "Failed to parse tuple." << std::endl;
				std::terminate();
			}

			tensorflow::GraphDef graph_def = _tuple.graph();
			tensorflow::Status tf_status = _session->Create(graph_def);
			if (!tf_status.ok()) {
				std::cout << "create graph has failed in line " << __LINE__ << " in file "
					<< __FILE__ << std::endl;
				std::terminate();
			}
			std::string init_name = _tuple.init_name();
			std::cout << init_name << std::endl;
			tf_status = _session->Run({}, {}, { init_name }, nullptr);
			if (!tf_status.ok()) {
				std::cout << "running init has  failed in line " << __LINE__
					<< " in file " << __FILE__ << std::endl;
				std::terminate();
			}
			std::vector<tensorflow::Tensor> var_init_values;
			std::vector<std::string> var_names;
			google::protobuf::Map<std::string, Names>& map_names =
				*_tuple.mutable_map_names();
			std::for_each(
				map_names.begin(), map_names.end(),
				[&var_names](google::protobuf::MapPair<std::string, Names>& pair) {
				var_names.push_back(pair.first);
				Names& names = pair.second;

				std::string assign_name_current = names.assign_name();
				size_t length = assign_name_current.length();
				*names.mutable_assign_name() =
					assign_name_current.substr(0, length - 2);
				std::cout << *names.mutable_assign_name() << std::endl;

				std::string assign_add_name_current = names.assign_add_name();
				length = assign_add_name_current.length();
				*names.mutable_assign_add_name() =
					assign_add_name_current.substr(0, length - 2);
			});
			tf_status = _session->Run({}, var_names, {}, &var_init_values);
			if (!tf_status.ok()) {
				std::cout << "getting init var value has failed in line " << __LINE__
					<< " in file " << __FILE__ << std::endl;
				std::terminate();
			}
			size_t size = var_names.size();
			for (size_t i = 0; i < size; i++) {
				tensorflow::TensorProto var_proto;
				var_init_values[i].AsProtoField(&var_proto);
				_tuple.mutable_map_parameters()->insert(
					google::protobuf::MapPair<std::string, tensorflow::TensorProto>(
						var_names[i], var_proto));
			}
			_tuple.set_interval(_interval);
			_tuple.set_lr(_lr);
			_tuple.set_total_iter(_total_iter);
			_init_time_point = std::chrono::high_resolution_clock::now();
			auto now = std::chrono::system_clock::now();
			auto init_time_t = std::chrono::system_clock::to_time_t(now);
			_label = std::to_string(init_time_t);
			std::string store_loss_file_path =
				"loss_result/adaptive" + _label +
				"_interval:" + std::to_string(_interval) +
				"_number_of_workers:" + std::to_string(_number_of_workers) + "_init_level:" +
				std::to_string(std::pow(2, static_cast<int>(_grad_quant_level)));
			_file_out_stream.open(store_loss_file_path);
			std::string store_state_file_path =
				"state_result/adaptive" + _label +
				"_interval:" + std::to_string(_interval) +
				"_number_of_workers:" + std::to_string(_number_of_workers) + "_init_level:" +
				std::to_string(std::pow(2, static_cast<int>(_grad_quant_level)));
			_file_state_stream.open(store_state_file_path);
			std::cout << "files opened" << std::endl;
		}

		Status retrieveTuple(ServerContext* context, const Empty* request,
			Tuple* reply) override {
			*reply = _tuple;
			return Status::OK;
		}

		Status sendLoss(::grpc::ServerContext* context,
			const ::adaptive_system::Loss* request,
			::adaptive_system::Empty* response) override {
			const float loss = request->loss();
			std::cout << "loss is " << loss << std::endl;
			std::unique_lock<std::mutex> lk(_mutex_loss);
			_bool_loss = false;
			_vector_loss.push_back(loss);
			if (_vector_loss.size() == _number_of_workers) {
				float sum =
					std::accumulate(_vector_loss.begin(), _vector_loss.end(), 0.0);
				float average = sum / _number_of_workers;
				_current_iter_number++;
				std::cout << "iteratino :" << _current_iter_number
					<< ", average loss is " << average << std::endl;
				auto now = std::chrono::high_resolution_clock::now();
				//std::time_t now_t = std::chrono::system_clock::to_time_t(now);
				//using seconds
				 std::chrono::duration<double> diff_time = (now - _init_time_point);
				_file_out_stream << std::to_string(diff_time.count())
				<< ":: iter num ::" << std::to_string(_current_iter_number)
					<< ":: loss is ::" << average << "\n";
				_file_out_stream.flush();
				_vector_loss_history.push_back(average);
				_vector_loss.clear();
				_bool_loss = true;
				_condition_variable_loss.notify_all();
			}
			else {
				_condition_variable_loss.wait(lk, [this] { return _bool_loss; });
			}
			lk.unlock();
			return Status::OK;
		}

		Status sendGradient(ServerContext* context, const NamedGradients* request,
			NamedGradients* response) override {
			NamedGradients& named_gradients = const_cast<NamedGradients&>(*request);
			std::map<std::string, tensorflow::Tensor> map_gradient;
			dequantize_gradient(named_gradients, map_gradient);
			std::unique_lock<std::mutex> lk(_mutex_gradient);
			_bool_gradient = false;
			_vector_map_gradient.push_back(
				map_gradient);  // result in copy which may slow down the process!!!!
			if (_vector_map_gradient.size() == _number_of_workers) {
				std::map<std::string, tensorflow::Tensor> map_gradient_another;
				aggregate_gradients(_vector_map_gradient, map_gradient_another);
				_store_named_gradient = NamedGradients();
				quantize_gradient(
					map_gradient_another, &_store_named_gradient,
					cast_grad_quant_level_to_quantization_type(_grad_quant_level));
				apply_quantized_gradient_to_model_using_adam(_store_named_gradient,
					_session, _tuple);
				_vector_map_gradient.clear();
				_bool_gradient = true;
				_condition_variable_gradient.notify_all();
			}
			else {
				_condition_variable_gradient.wait(lk, [this] { return _bool_gradient; });
			}
			lk.unlock();
			*response = _store_named_gradient;
			return Status::OK;
		}

		Status sendState(ServerContext* context, const PartialState* request,
			QuantizationLevel* response) override {
			std::unique_lock<std::mutex> lk(_mutex_state);
			_bool_state = false;
			_vector_partial_state.push_back(*request);
			if (_vector_partial_state.size() == _number_of_workers) {
				if (_bool_is_first_iteration) {
					_bool_is_first_iteration = false;
					_last_state = get_final_state_from_partial_state(_vector_partial_state);
					print_state_to_file(_last_state);
					//need not to store last action because _grad_quant_level can represent it
					if (_vector_loss_history.size() != 1) {
						PRINT_ERROR_MESSAGE("when in first iteration, the _vector_loss_history's size must be 1");
						std::terminate();
					}
					_last_loss = _vector_loss_history[0];
					_vector_loss_history.clear();
					_time_point_last = std::chrono::high_resolution_clock::now();
				}
				else {
					adjust_rl_model(_vector_partial_state);
				}
				_vector_partial_state.clear();
				_bool_state = true;
				_condition_variable_state.notify_all();
				std::cout << "got line " << __LINE__ << std::endl;
			}
			else {
				std::cout << "got line " << __LINE__ << std::endl;
				_condition_variable_state.wait(lk, [this] { return _bool_state; });
				std::cout << "got line " << __LINE__ << std::endl;
			}
			lk.unlock();
			response->set_level(_grad_quant_level);
			return Status::OK;
		}

		// private member functions
	private:
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
					}
					else {
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

		
		void adjust_rl_model(std::vector<PartialState> const& vector_partial_state) {
			tensorflow::Tensor state_tensor = get_final_state_from_partial_state(vector_partial_state);
			print_state_to_file(state_tensor);
			GRAD_QUANT_LEVEL new_action = _sarsa.sample_new_action(state_tensor);
			GRAD_QUANT_LEVEL old_action = _grad_quant_level;
			auto now_time_point = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> diff = now_time_point - _time_point_last;
			float diff_seconds = diff.count();
			auto loss_sum = std::accumulate(_vector_loss_history.begin(), _vector_loss_history.end(), 0.0f);
			auto average = loss_sum / _interval;
			float reward = (_last_loss - average) / diff_seconds;
			_sarsa.adjust_model(reward, _last_state, old_action, state_tensor, new_action);
			_grad_quant_level = new_action;
			std::cout << "diff_seconds is: " << diff_seconds << " reward is " << reward
				<< " quantization level become: " << std::pow(2, static_cast<int>(_grad_quant_level)) << std::endl;
			_vector_loss_history.clear();
			_last_loss = average;
			_last_state = state_tensor;
			_time_point_last = std::chrono::high_resolution_clock::now();
		}
		
		// private data member
	private:
		const int _interval;
		const float _lr;
		const int _total_iter;
		const int _number_of_workers;
		int _current_iter_number = 0;
		GRAD_QUANT_LEVEL _grad_quant_level = GRAD_QUANT_LEVEL::NONE;

		std::chrono::time_point<std::chrono::high_resolution_clock> _init_time_point;
		std::chrono::time_point<std::chrono::high_resolution_clock> _time_point_last;


		std::mutex _mutex_gradient;
		std::mutex _mutex_state;
		std::mutex _mutex_loss;
		std::condition_variable _condition_variable_gradient;
		std::condition_variable _condition_variable_state;
		std::condition_variable _condition_variable_loss;
		std::vector<std::map<std::string, tensorflow::Tensor>> _vector_map_gradient;
		std::vector<PartialState> _vector_partial_state;
		float _last_loss;
		std::vector<float> _vector_loss;
		std::vector<float> _vector_loss_history;
		std::string _tuple_local_path;
		bool _bool_gradient;
		bool _bool_state;
		bool _bool_loss;
		bool _bool_is_first_iteration = true;
		NamedGradients _store_named_gradient;

		tensorflow::Session* _session;
		Tuple _tuple;
		std::ofstream _file_out_stream;
		std::ofstream _file_state_stream;

		sarsa_model _sarsa;
		tensorflow::Tensor _last_state;
		std::string _label;
	};
}
adaptive_system::GRAD_QUANT_LEVEL cast_int_to_grad_quant_level(int level) {
	switch (level) {
	case 1:
		return adaptive_system::GRAD_QUANT_LEVEL::ONE;
	case 2:
		return adaptive_system::GRAD_QUANT_LEVEL::TWO;
	case 4:
		return adaptive_system::GRAD_QUANT_LEVEL::FOUR;
	case 8:
		return adaptive_system::GRAD_QUANT_LEVEL::EIGHT;
	case 16:
		return adaptive_system::GRAD_QUANT_LEVEL::SIXTEEN;
	default:
		return adaptive_system::GRAD_QUANT_LEVEL::NONE;
	}
}

int main(int argc, char** argv) {
	std::string server_address("0.0.0.0:50051");
	int interval = atoi(argv[1]);
	float learning_rate = atof(argv[2]);
	int total_iter = atoi(argv[3]);
	int number_of_workers = atoi(argv[4]);
	int level = atoi(argv[5]);
	std::string tuple_path = argv[6];
	std::string sarsa_path = argv[7];
	float r = atof(argv[8]);
	float eps_greedy = atof(argv[9]);

	adaptive_system::RPCServiceImpl service(
		interval, learning_rate, total_iter, number_of_workers,
		cast_int_to_grad_quant_level(level), tuple_path, sarsa_path, r, eps_greedy);

	ServerBuilder builder;
	// Listen on the given address without any authentication mechanism.
	builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
	builder.SetMaxMessageSize(std::numeric_limits<int>::max());
	// Register "service" as the instance through which we'll communicate with
	// clients. In this case it corresponds to an *synchronous* service.
	builder.RegisterService(&service);
	// Finally assemble the server.
	std::unique_ptr<Server> server(builder.BuildAndStart());
	std::cout << "Server listening on " << server_address << std::endl;

	// Wait for the server to shutdown. Note that some other thread must be
	// responsible for shutting down the server for this call to ever return.
	server->Wait();

	return 0;
}
