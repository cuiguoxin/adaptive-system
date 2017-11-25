#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <numeric>
#include <thread>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#include "quantization/util/any_level.h"
#include "quantization/util/helper.h"
#include "quantization/util/algorithms.h"
#include "server/sarsa.h"
#include "server/reward.h"

namespace input {
	using namespace tensorflow;

	unsigned int index_current = 0;
	int const batch_size = 32;
	std::vector<Tensor> raw_tensors, standard_images, standard_labels;
	const int record_size = 3073;
	const int label_size = 1;
	const int image_size = 3072;

	namespace {
		Session* load_graph_and_create_session(const std::string& graph_path) {
			GraphDef graph_def;
			Status status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
			if (!status.ok()) {
				std::cout << status.ToString() << "\n";
				std::terminate();
			}
			Session* session;
			status = NewSession(SessionOptions(), &session);
			if (!status.ok()) {
				std::cout << status.ToString() << "\n";
				std::terminate();
			}
			status = session->Create(graph_def);
			if (!status.ok()) {
				std::cout << status.ToString() << "\n";
				std::terminate();
			}
			return session;
		}
		void read_raw_tensors_from_file(const std::string& binary_file_prefix) {
			for (int i = 1; i <= 5; i++) {
				std::ifstream input_stream(
					binary_file_prefix + std::to_string(i) + ".bin", std::ios::binary);
				TensorShape raw_tensor_shape({ record_size });
				if (input_stream.is_open()) {
					for (int j = 0; j < 10000; j++) {
						Tensor raw_tensor(DataType::DT_UINT8, raw_tensor_shape);
						uint8* raw_tensor_ptr = raw_tensor.flat<uint8>().data();
						input_stream.read(reinterpret_cast<char*>(raw_tensor_ptr), record_size);
						raw_tensors.push_back(raw_tensor);
					}
				}
				input_stream.close();
			}
			PRINT_INFO;
			// shuffle the vector raw_tensors
			/*unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::shuffle(raw_tensors.begin(), raw_tensors.end(),
				std::default_random_engine(seed));*/
		}
	}

	void turn_raw_tensors_to_standard_version(
		const std::string& binary_file_prefix
		="/home/cgx/git_project/adaptive-system/resources/cifar-10-batches-bin/data_batch_",
		const std::string& preprocess_graph_path
		= "/home/cgx/git_project/adaptive-system/input/cifar10/preprocess.pb") {
		PRINT_INFO;
		Session* session = load_graph_and_create_session(preprocess_graph_path);
		PRINT_INFO;
		read_raw_tensors_from_file(binary_file_prefix);
		std::cout << raw_tensors.size() << std::endl;
		for (int i = 0; i < 50000; i++) {
			Tensor raw_tensor = raw_tensors[i];
			std::vector<Tensor> image_and_label;
			Status status = session->Run({ { "raw_tensor", raw_tensor } }, { "div", "label" },
			{}, &image_and_label);
			if (!status.ok()) {
				std::cout << "failed in line " << __LINE__ << " in file " << __FILE__
					<< " " << status.error_message() << std::endl;
				std::terminate();
			}
			standard_images.push_back(image_and_label[0]);
			standard_labels.push_back(image_and_label[1]);
		}
		raw_tensors.clear();
		PRINT_INFO;
	}
	std::pair<Tensor, Tensor> get_next_batch() {
		static std::mutex mu;
		int standard_images_size = 3 * 28 * 28;
		TensorShape images_batch_shape({ batch_size, 28, 28, 3 }),
			labels_batch_shape({ batch_size });
		Tensor images_batch(DataType::DT_FLOAT, images_batch_shape),
			labels_batch(DataType::DT_INT32, labels_batch_shape);
		float* images_batch_ptr = images_batch.flat<float>().data();
		int* label_batch_ptr = labels_batch.flat<int>().data();
		std::unique_lock<std::mutex> lk(mu);
		for (int i = 0; i < batch_size; i++) {
			int real_index = index_current % 50000;
			Tensor& image_current = standard_images[real_index];
			float* image_current_ptr = image_current.flat<float>().data();
			std::copy(image_current_ptr, image_current_ptr + standard_images_size,
				images_batch_ptr + i * standard_images_size);
			Tensor& label_current = standard_labels[real_index];
			int* label_current_ptr = label_current.flat<int>().data();
			label_batch_ptr[i] = *label_current_ptr;
			index_current++;
		}
		lk.unlock();
		return std::pair<Tensor, Tensor>(images_batch, labels_batch);
	}
}

namespace client {

	using namespace tensorflow;
	using namespace adaptive_system;

	Tuple tuple;
	Session* session;
	std::string batch_placeholder_name;
	std::string label_placeholder_name;
	const int threshold_to_quantize = 105000;

	namespace {
		std::pair<float, float> compute_gradient_and_loss(
			std::vector<std::pair<std::string, tensorflow::Tensor>> feeds,
			std::map<std::string, tensorflow::Tensor>& gradients) {

			std::vector<std::string> fetch;
			std::string loss_name = tuple.loss_name();
			auto entropy_loss = tuple.cross_entropy_loss_name();
			fetch.push_back(loss_name);
			fetch.push_back(entropy_loss);
			std::vector<tensorflow::Tensor> outputs;
			std::vector<std::string> variable_names_in_order;
			google::protobuf::Map<std::string, Names> const& map_names =
				tuple.map_names();
			std::for_each(map_names.begin(), map_names.end(),
				[&fetch, &variable_names_in_order](
					google::protobuf::MapPair<std::string, Names> const& pair) {
				Names const& names = pair.second;
				std::string const& variable_name = pair.first;
				fetch.push_back(names.gradient_name());
				variable_names_in_order.push_back(variable_name);
			});
			tensorflow::Status tf_status = session->Run(feeds, fetch, {}, &outputs);
			if (!tf_status.ok()) {
				PRINT_ERROR_MESSAGE(tf_status.error_message());
				std::terminate();
			}
			tensorflow::Tensor& loss_tensor = outputs[0];
			float* loss_ptr = loss_tensor.flat<float>().data();
			float loss_ret = loss_ptr[0];

			auto& entropy_tensor = outputs[1];
			float* entropy_ptr = entropy_tensor.flat<float>().data();
			float loss_entropy_ret = entropy_ptr[0];

			outputs.erase(outputs.begin());
			outputs.erase(outputs.begin());

			size_t size = outputs.size();
			for (size_t i = 0; i < size; i++) {
				gradients.insert(std::pair<std::string, tensorflow::Tensor>(
					variable_names_in_order[i], outputs[i]));
			}
			return { loss_ret, loss_entropy_ret };
		}
	}

	void load_primary_model_and_init() {
		const std::string tuple_local_path = 
			"/home/cgx/git_project/adaptive-system/input/cifar10/tuple_gradient_descent.pb";
		session = tensorflow::NewSession(tensorflow::SessionOptions());
		std::fstream input(tuple_local_path, std::ios::in | std::ios::binary);

		if (!input) {
			std::cout << tuple_local_path
				<< ": File not found.  Creating a new file." << std::endl;
		}
		else if (!tuple.ParseFromIstream(&input)) {
			std::cerr << "Failed to parse tuple." << std::endl;
			std::terminate();
		}
		input.close();
		GraphDef graph_def = tuple.graph();
		Status tf_status = session->Create(graph_def);
		if (!tf_status.ok()) {
			std::cout << "create graph has failed in line " << __LINE__ << " in file "
				<< __FILE__ << std::endl;
			std::terminate();
		}

		//init parameters
		std::string init_name = tuple.init_name();
		std::cout << init_name << std::endl;
		tf_status = session->Run({}, {}, { init_name }, nullptr);
		if (!tf_status.ok()) {
			std::cout << "running init has  failed in line " << __LINE__
				<< " in file " << __FILE__ << std::endl;
			std::terminate();
		}
		//init some names
		batch_placeholder_name = tuple.batch_placeholder_name();
		label_placeholder_name = tuple.label_placeholder_name();
		
	}

	void compute_gradient_loss_and_quantize(const int level,
		std::map<std::string, tensorflow::Tensor>& map_gradients, std::pair<float, float>& loss) {
		//PRINT_INFO;
		std::pair<tensorflow::Tensor, tensorflow::Tensor> feeds =
			input::get_next_batch();
		//PRINT_INFO;
		loss = compute_gradient_and_loss(
		{ { batch_placeholder_name, feeds.first },
		{ label_placeholder_name, feeds.second } },
			map_gradients);
		//PRINT_INFO;
		NamedGradientsAccordingColumn named_gradients_send;
		quantize_gradients_according_column(map_gradients,
			&named_gradients_send, level, threshold_to_quantize);
		map_gradients.clear();
		dequantize_gradients_according_column(named_gradients_send, map_gradients);
	}

	namespace log {

		std::ofstream file_loss_stream;

		void init_log(int const interval, 
			int const total_worker_num,
			int const start_level, 
			int const end_level) {
			//init log
			auto now = std::chrono::system_clock::now();
			auto init_time_t = std::chrono::system_clock::to_time_t(now);
			std::string label = std::to_string(init_time_t);
			std::string store_loss_file_path =
				"loss_result/sarsa_adaptive" + label +
				"_interval:" + std::to_string(interval) +
				"_number_of_workers:" + std::to_string(total_worker_num)
				+ "_level" + std::to_string(start_level) + "-" + std::to_string(end_level)
				;
			file_loss_stream.open(store_loss_file_path);
		}
		inline void log(float const time,
			float const loss,
			float const cross_entropy_loss,
			int const current_iter,
			int const current_level) {
			file_loss_stream << std::to_string(time)
				<< ":: iter num ::" << std::to_string(current_iter)
				<< ":: loss is ::" << loss << "::" << current_level <<
				":: cross_entropy_loss is :: " << cross_entropy_loss << "\n";
			file_loss_stream.flush();
		}
	}

	namespace sarsa {

		tensorflow::Tensor last_state;
		std::vector<float> loss_history;
		float _last_loss = 16.0f;

		void adjust_rl_model(sarsa_model& sm, int& level) {
			std::vector<float> moving_average_losses;
			const float r = 0.98;
			_last_loss = moving_average_from_last_loss(_last_loss,
				loss_history,
				moving_average_losses, r);
			tensorflow::Tensor new_state = get_float_tensor_from_vector(moving_average_losses);
			int new_level = sm.sample_new_action(new_state);
			int old_level = level;

			float slope = get_slope_according_loss(moving_average_losses);
			std::cout << "slope is " << slope << " new level is " << new_level << std::endl;
			float reward = get_reward_v5(slope, old_level); // -slope * 100 / level
			sm.adjust_model(reward, last_state, old_level, new_state, new_level);
			level = new_level;
			last_state = new_state;
			loss_history.clear();
		}
	}

	void do_work(int const total_iter_num,
		int const total_worker_num,
		int const init_level,
		int const interval, 
		int const start_level, 
		int const end_level,
		float const eps_greedy, 
		float const loss_threshold) {
		//init sarsa
		PRINT_INFO;
		sarsa_model sm("/home/cgx/git_project/adaptive-system/single_implement/sarsa_continous.pb",
			interval, 0.8, eps_greedy, start_level, end_level, init_level);
		PRINT_INFO;
		sarsa::last_state = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT,
			tensorflow::TensorShape({interval}));
		float* ptr_last_state = sarsa::last_state.flat<float>().data();
		std::fill(ptr_last_state, ptr_last_state + interval, 16.0f);
		//init log
		log::init_log(interval, total_worker_num, start_level, end_level);
		load_primary_model_and_init();
		int level = init_level;
		for (int i = 0; i < total_iter_num; i++) {
			std::vector<std::map<std::string, tensorflow::Tensor>> vec_grads;
			std::vector<std::thread> vec_threads;
			std::vector<std::pair<float, float>> vec_losses;
			vec_losses.resize(total_worker_num);
			vec_grads.resize(total_worker_num);
			for (int j = 0; j < total_worker_num; j++) {
				vec_threads.push_back(
					std::thread(
						compute_gradient_loss_and_quantize,
						level,
						std::ref(vec_grads[j]),
						std::ref(vec_losses[j])));
			}
			for (int j = 0; j < total_worker_num; j++) {
				vec_threads[j].join();
			}
			PRINT_INFO;
			//finished gradient computing
			std::map<std::string, tensorflow::Tensor> merged_gradient;
			aggregate_gradients(vec_grads, merged_gradient);
			PRINT_INFO;
			average_gradients(total_worker_num, merged_gradient);
			NamedGradientsAccordingColumn store_named_gradient;
			PRINT_INFO;
			quantize_gradients_according_column(
				merged_gradient, &store_named_gradient,
				level, threshold_to_quantize);
			PRINT_INFO;
			apply_quantized_gradient_to_model(store_named_gradient,
				session, tuple);
			//log
			std::vector<float> total_losses;
			std::vector<float> entropy_losses;
			for (auto pair : vec_losses) {
				total_losses.push_back(pair.first);
				entropy_losses.push_back(pair.second);
			}
			float const average = std::accumulate(
				total_losses.begin(), 
				total_losses.end(),
				0.0f) / total_worker_num;
			float const entropy_average = std::accumulate(
				entropy_losses.begin(),
				entropy_losses.end(),
				0.0f) / total_worker_num;
			log::log(0, average, entropy_average, i, level);
			std::cout << "iter num is : " << i << " "
				<< " loss is : " << average << " , cross_entropy loss is "
				<< entropy_average << " level is " << level << std::endl;
			
			//check if it's time to change level
			const int start_iter_num = 6;
			int real_num = i - start_iter_num;
			if (real_num <= 0) {
				sarsa::_last_loss = average;
				continue;
			}
			//add average to loss_history
			sarsa::loss_history.push_back(average);

			if (real_num % interval == 0) {
				sarsa::adjust_rl_model(sm, level);
			}
			if (average < loss_threshold) {
				level = 4;
			}
		}
	}

}

int main(int argc, char** argv) {
	int const total_iter_num = atoi(argv[1]);
	int const total_worker_num = atoi(argv[2]);
	int const init_level = atoi(argv[3]);
	int const interval = atoi(argv[4]);
	int const start_level = atoi(argv[5]);
	int const end_level = atoi(argv[6]);
	float const eps_greedy = atof(argv[7]);
	float const loss_threshold = atof(argv[8]);
	PRINT_INFO;
	input::turn_raw_tensors_to_standard_version();
	client::do_work(total_iter_num,
		total_worker_num,
		init_level,
		interval,
		start_level,
		end_level,
		eps_greedy, 
		loss_threshold);

	return 0;
}
