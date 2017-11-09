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
#include <cstdlib>

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

namespace input {
	using namespace tensorflow;

	unsigned int index_current = 0;
	int batch_size = 0;
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
			
			// shuffle the vector raw_tensors
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::shuffle(raw_tensors.begin(), raw_tensors.end(),
				std::default_random_engine(seed));
		}
	}

	void turn_raw_tensors_to_standard_version(
		const std::string& binary_file_prefix
		= "/home/cgx/git_project/adaptive-system/resources/cifar-10-batches-bin/data_batch_",
		const std::string& preprocess_graph_path
		= "/home/cgx/git_project/adaptive-system/input/cifar10/preprocess.pb") {
		
		Session* session = load_graph_and_create_session(preprocess_graph_path);
		
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
		float one_iteration(
			std::vector<std::pair<std::string, tensorflow::Tensor>> const & feeds) {

			std::vector<std::string> fetch;
			std::string loss_name = tuple.loss_name();
			fetch.push_back(loss_name);
			std::vector<tensorflow::Tensor> outputs;
			std::vector<std::string> no_fetch;
			no_fetch.push_back(tuple.training_op_name());
			
			tensorflow::Status tf_status = session->Run(feeds, fetch, no_fetch, &outputs);
			if (!tf_status.ok()) {
				PRINT_ERROR_MESSAGE(tf_status.error_message());
				std::terminate();
			}
			tensorflow::Tensor& loss_tensor = outputs[0];
			float* loss_ptr = loss_tensor.flat<float>().data();
			float loss_ret = loss_ptr[0];
			
			return loss_ret;
		}
	}

	void load_primary_model_and_init(int const batch_size) {
		const std::string tuple_local_path =
			"/home/cgx/git_project/adaptive-system/single_implement/no_quantization/tuple_gradient_descent.pb";
		std::string command = "python test_gradient_descent.py " + std::to_string(batch_size);
		int err_code = system(command.c_str());
		if (err_code != 0) {
			std::cout << "error happens in line " << __LINE__ << std::endl;
			std::terminate();
		}
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

	

	void do_work(int const total_iter_num, int const batch_size) {

		load_primary_model_and_init(batch_size);
		auto pic_name = tuple.batch_placeholder_name();
		auto lab_name = tuple.label_placeholder_name();

		for (int i = 0; i < total_iter_num; i++) {
			auto batch = input::get_next_batch();

			float loss = one_iteration({ {pic_name, batch.first},{lab_name, batch.second} });
			std::cout << "iter num: " << i << ", loss: " << loss << std::endl;
		}
	}

}

int main(int argc, char** argv) {
	int const total_iter_num = atoi(argv[1]);
	input::batch_size = atoi(argv[2]);
	input::turn_raw_tensors_to_standard_version();
	client::do_work(total_iter_num, input::batch_size);

	return 0;
}
