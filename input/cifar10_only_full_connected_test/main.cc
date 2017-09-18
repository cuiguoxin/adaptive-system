#include "input/cifar10/input.h"
#include "quantization/util/helper.h"
#include "proto/rpc_service.pb.h"
#include "quantization/util/any_level.h"
#include <cstdlib>
#include <fstream>

using namespace adaptive_system;

tensorflow::Tensor quantize_then_dequantize(int const level,
	tensorflow::Tensor const & origin_tensor) {
	int size = origin_tensor.NumElements();
	if (size < 1024 * 101) {
		return origin_tensor;
	}
	GradientAccordingColumn gradient;
	tensorflow::Tensor return_tensor;
	quantize_gradient_according_column(level, origin_tensor, gradient);
	dequantize_gradient_according_column(gradient, return_tensor);
	return return_tensor;
}

void train(Tuple const & tuple,
	tensorflow::Session* session,
	int const pre_level, int const split_iter, int const post_level, 
	int const batch_size, float const lr) {
	auto now = std::chrono::system_clock::now();
	auto init_time_t = std::chrono::system_clock::to_time_t(now);
	auto label = std::to_string(init_time_t);
	std::string loss_file_name = "/home/cgx/git_project/adaptive-system/input/"
		"cifar10_only_full_connected_test/log/loss" + label +
		"_level_" + std::to_string(pre_level) + "-" +
		std::to_string(split_iter) + "-" + std::to_string(post_level)+ "_"
		+ "_batch_size_" + std::to_string(batch_size)
		+ "_lr_" + std::to_string(lr);
	std::ofstream loss_stream(loss_file_name);
	auto image_ph_name = tuple.batch_placeholder_name();
	auto label_ph_name = tuple.label_placeholder_name();
	auto train_op_name = tuple.training_op_name();
	auto loss_name = tuple.loss_name();
	auto& map_names = tuple.map_names();
	PRINT_INFO;
	std::vector<std::string> fetch, gradient_names;
	for (auto& pair : map_names) {
		gradient_names.push_back(pair.second.gradient_name());
		fetch.push_back(pair.second.gradient_name());
		//std::cout << pair.second.gradient_name() << std::endl;
	}
	fetch.push_back(loss_name);
	int const iteration_num = 2000;

	std::string const raw_binary_path = "/home/cgx/git_project/adaptive-system/resources/cifar-10-batches-bin/data_batch_1.bin";
	std::string const preprocess_path = "/home/cgx/git_project/adaptive-system/input/cifar10/preprocess.pb";
	cifar10::turn_raw_tensors_to_standard_version(raw_binary_path, preprocess_path);

	for (int i = 0; i < iteration_num; i++) {
		auto train_pair = cifar10::get_next_batch(batch_size);
		auto& image_tensor = train_pair.first;
		auto& label_tensor = train_pair.second;
		std::vector<tensorflow::Tensor> result;
		tensorflow::Status tf_status = session->Run(
		{ { image_ph_name, image_tensor },{ label_ph_name, label_tensor } },
			fetch, {}, &result);
		if (!tf_status.ok()) {
			PRINT_ERROR_MESSAGE(tf_status.error_message());
			std::terminate();
		}
		int size = result.size();
		float* loss_ptr = result[size - 1].flat<float>().data();
		std::cout << "loss is : " << loss_ptr[0] << std::endl;
		loss_stream << "loss is : " << loss_ptr[0] << std::endl;
		result.resize(size - 1);
		std::vector<std::pair<std::string, tensorflow::Tensor>> feed;
		int level = pre_level;
		if (i > split_iter) {
			level = post_level;
		}
		for (int j = 0; j < size - 1; j++) {
			tensorflow::Tensor tensor_feed = quantize_then_dequantize(level, result[j]);
			feed.push_back(std::pair<std::string, tensorflow::Tensor>(gradient_names[j], tensor_feed));
		}
		tf_status = session->Run(feed, {}, { train_op_name }, nullptr);
		if (!tf_status.ok()) {
			PRINT_ERROR_MESSAGE(tf_status.error_message());
			std::terminate();
		}

	}
}

int main(int argc, char* argv[]) {
	int const pre_level = atoi(argv[1]);
	int const split_iter = atoi(argv[2]);
	int const post_level = atoi(argv[3]);
	int const batch_size = atoi(argv[4]);
	float const lr = atof(argv[5]);
	std::string command = "python create_primary_model_using_adam.py " + std::to_string(lr) + " " + std::to_string(batch_size);
	int code = system(command.c_str());
	if (code != 0) {
		PRINT_ERROR_MESSAGE("failed in generate tuple file, error code is " + std::to_string(code));
		std::terminate();
	}
	using namespace adaptive_system;
	Tuple tuple;
	std::string tuple_path = "/home/cgx/git_project/adaptive-system/input/"
		"cifar10_only_full_connected_test/tuple_adam_test.pb";
	tensorflow::Session* session = tensorflow::NewSession(tensorflow::SessionOptions());
	std::fstream input(tuple_path, std::ios::in | std::ios::binary);
	if (!input) {
		std::cout << tuple_path
			<< ": File not found.  Creating a new file." << std::endl;
	}
	else if (!tuple.ParseFromIstream(&input)) {
		std::cerr << "Failed to parse tuple." << std::endl;
		std::terminate();
	}

	tensorflow::GraphDef graph_def = tuple.graph();
	tensorflow::Status tf_status = session->Create(graph_def);
	if (!tf_status.ok()) {
		PRINT_ERROR_MESSAGE(tf_status.error_message());
		std::terminate();
	}
	std::string init_name = tuple.init_name();
	std::cout << init_name << std::endl;
	tf_status = session->Run({}, {}, { init_name }, nullptr);
	if (!tf_status.ok()) {
		PRINT_ERROR_MESSAGE(tf_status.error_message());
		std::terminate();
	}
	PRINT_INFO;
	train(tuple, session, pre_level, split_iter, post_level, batch_size, lr);

	return 0;
}
