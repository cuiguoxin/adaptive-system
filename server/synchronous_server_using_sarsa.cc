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
#include "server/reward.h"

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
			int init_level,
			std::string const& tuple_local_path,
			std::string const& sarsa_model_path,
			int const threshold,
			int const sarsa_model_input_size)
			: SystemControl::Service(),
			_interval(interval),
			_lr(lr),
			_total_iter(total_iter),
			_number_of_workers(number_of_workers),
			_level(init_level),
			_tuple_local_path(tuple_local_path),
			_sarsa(sarsa_model_path, sarsa_model_input_size, 0.9, 0.1, 1, 8, init_level), 
			_threshold(threshold)
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

			//init parameters
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

			});
			PRINT_INFO;
			tf_status = _session->Run({}, var_names, {}, &var_init_values);
			if (!tf_status.ok()) {
				std::cout << "getting init var value has failed in line " << __LINE__
					<< " in file " << __FILE__ << std::endl;
				std::terminate();
			}
			PRINT_INFO;
			size_t size = var_names.size();
			for (size_t i = 0; i < size; i++) {
				tensorflow::TensorProto var_proto;
				var_init_values[i].AsProtoField(&var_proto);
				_tuple.mutable_map_parameters()->insert(
					google::protobuf::MapPair<std::string, tensorflow::TensorProto>(
						var_names[i], var_proto));
			}
			PRINT_INFO;
			_tuple.set_interval(_interval);
			_tuple.set_lr(_lr);
			_tuple.set_total_iter(_total_iter);
			_tuple.set_threshold_to_quantize(_threshold);
			_init_time_point = std::chrono::high_resolution_clock::now();
			auto now = std::chrono::system_clock::now();
			auto init_time_t = std::chrono::system_clock::to_time_t(now);
			_label = std::to_string(init_time_t);
			std::string store_loss_file_path =
				"loss_result/sarsa_adaptive" + _label +
				"_interval:" + std::to_string(_interval) +
				"_number_of_workers:" + std::to_string(_number_of_workers)
				;
			_file_loss_stream.open(store_loss_file_path);
			std::string store_state_file_path =
				"state_result/sarsa_adaptive" + _label +
				"_interval:" + std::to_string(_interval) +
				"_number_of_workers:" + std::to_string(_number_of_workers)
				;
			_file_state_stream.open(store_state_file_path);
			std::string store_action_file_path =
				"action_result/sarsa_adaptive" + _label +
				"_interval:" + std::to_string(_interval) +
				"_number_of_workers:" + std::to_string(_number_of_workers)
				;
			_file_action_stream.open(store_action_file_path);
			std::cout << "files opened" << std::endl;
			PRINT_INFO;
		}

		grpc::Status retrieveTuple(ServerContext* context, const Empty* request,
			Tuple* reply) override {
			*reply = _tuple;
			_mutex_tuple.lock();
			_count++;
			if (_count == _number_of_workers) {
				_tuple.mutable_map_parameters()->clear();
			}
			_mutex_tuple.unlock();
			return grpc::Status::OK;
		}

		grpc::Status sendLoss(::grpc::ServerContext* context,
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
				std::chrono::duration<double> diff_time = (now - _init_time_point);
				_vector_time_history.push_back(diff_time.count());
				_file_loss_stream << std::to_string(diff_time.count())
					<< ":: iter num ::" << std::to_string(_current_iter_number)
					<< ":: loss is ::" << average << "::" << _level <<  "\n";
				_file_loss_stream.flush();
				_vector_loss_history.push_back(average);
				_vector_loss.clear();
				_bool_loss = true;
				_condition_variable_loss.notify_all();
			}
			else {
				_condition_variable_loss.wait(lk, [this] { return _bool_loss; });
			}
			lk.unlock();
			return grpc::Status::OK;
		}

		grpc::Status sendGradient(ServerContext* context, const NamedGradientsAccordingColumn* request,
			NamedGradientsAccordingColumn* response) override {
			NamedGradientsAccordingColumn& named_gradients 
				= const_cast<NamedGradientsAccordingColumn&>(*request);
			std::map<std::string, tensorflow::Tensor> map_gradient;
			PRINT_INFO;
			dequantize_gradients_according_column(named_gradients, map_gradient);
			std::unique_lock<std::mutex> lk(_mutex_gradient);
			_bool_gradient = false;
			_vector_map_gradient.push_back(
				map_gradient);  // result in copy which may slow down the process!!!!
			if (_vector_map_gradient.size() == _number_of_workers) {
				std::map<std::string, tensorflow::Tensor> merged_gradient;
				PRINT_INFO;
				aggregate_gradients(_vector_map_gradient, merged_gradient);
				PRINT_INFO;
				average_gradients(_number_of_workers, merged_gradient);
				_store_named_gradient = NamedGradientsAccordingColumn();
				PRINT_INFO;
				quantize_gradients_according_column(
					merged_gradient, &_store_named_gradient,
					_level, _threshold);
				PRINT_INFO;
				apply_quantized_gradient_to_model(_store_named_gradient,
					_session, _tuple);
				PRINT_INFO;
				_vector_map_gradient.clear();

				_bool_gradient = true;
				_condition_variable_gradient.notify_all();
			}
			else {
				_condition_variable_gradient.wait(lk, [this] { return _bool_gradient; });
			}
			lk.unlock();
			*response = _store_named_gradient;
			return grpc::Status::OK;
		}

		grpc::Status sendState(ServerContext* context, const PartialState* request,
			QuantizationLevel* response) override {
			std::unique_lock<std::mutex> lk(_mutex_state);
			_bool_state = false;
			_vector_partial_state.push_back(*request);
			if (_vector_partial_state.size() == _number_of_workers) {
				if (_bool_is_first_iteration) {
					_bool_is_first_iteration = false;
					std::vector<float> temp_vector_loss;
					temp_vector_loss.resize(_interval, _vector_loss_history[0]);
					_last_state = get_float_tensor_from_vector(temp_vector_loss);
					print_state_to_file(_last_state);
					//need not to store last action because _grad_quant_level can represent it
					if (_vector_loss_history.size() != 1) {
						PRINT_ERROR_MESSAGE("when in first iteration, the _vector_loss_history's size must be 1");
						std::terminate();
					}

					_vector_loss_history.clear();
					_vector_time_history.clear();
				}
				else {
					adjust_rl_model();
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
			response->set_level_order(_level);

			return grpc::Status::OK;
		}

		// private member functions
	private:

		void adjust_rl_model() {
			std::vector<float> moving_average_losses;
			const float r = 0.9;
			moving_average_with_minus_average(
				_vector_loss_history,
				moving_average_losses, r);
			tensorflow::Tensor new_state = get_float_tensor_from_vector(moving_average_losses);
			int new_level = _sarsa.sample_new_action(new_state);
			int old_level = _level;

			standard_times(_vector_time_history);
			float slope = get_slope_according_loss(moving_average_losses);
			std::cout << "slope is " << slope << " new level is " << new_level << std::endl;
			float reward = get_reward_v4(slope, old_level); // -slope * 100 / level
			_sarsa.adjust_model(reward, _last_state, old_level, new_state, new_level);
			_level = new_level;

			_file_action_stream << std::to_string(_current_iter_number) << "::"
				<< std::to_string(new_level) << "\n";
			_file_action_stream.flush();
			_vector_loss_history.clear();
			_vector_time_history.clear();
			_last_state = new_state;
		}


		// private data member
	private:
		const int _interval;
		const float _lr;
		const int _total_iter;
		const int _number_of_workers;
		int _current_iter_number = 0;
		int _level = 0;
		const int _threshold;

		std::chrono::time_point<std::chrono::high_resolution_clock> _init_time_point;
		std::chrono::time_point<std::chrono::high_resolution_clock> _time_point_last;

		int _count = 0;
		std::mutex _mutex_tuple;
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
		std::vector<float> _vector_time_history;
		std::vector<float> _vector_loss_history;
		std::string _tuple_local_path;
		bool _bool_gradient;
		bool _bool_state;
		bool _bool_loss;
		bool _bool_is_first_iteration = true;
		NamedGradientsAccordingColumn _store_named_gradient;

		tensorflow::Session* _session;
		Tuple _tuple;
		std::ofstream _file_loss_stream;
		std::ofstream _file_action_stream;
		std::ofstream _file_state_stream;
		tensorflow::Tensor _last_state;
		sarsa_model _sarsa;
		std::string _label;
	};
}

int main(int argc, char** argv) {
	std::string server_address("0.0.0.0:50051");
	int interval = atoi(argv[1]);
	float learning_rate = atof(argv[2]);
	int total_iter = atoi(argv[3]);
	int number_of_workers = atoi(argv[4]);
	int init_level = atoi(argv[5]);
	std::string tuple_path = argv[6];
	std::string sarsa_path = argv[7];
	int threshold = atoi(argv[8]);
	int sarsa_model_input_size = atoi(argv[9]);

	adaptive_system::RPCServiceImpl service(
		interval, learning_rate, total_iter, number_of_workers,
		init_level, tuple_path, sarsa_path, threshold, sarsa_model_input_size);

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
