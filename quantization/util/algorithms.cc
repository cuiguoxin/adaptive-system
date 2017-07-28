#include "quantization/util/algorithms.h"
#include "quantization/util/helper.h"

#include <sstream>
#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

namespace adaptive_system {

	void apply_quantized_gradient_to_model(NamedGradients& named_gradients,
		tensorflow::Session* sess,
		Tuple& tuple) {
		google::protobuf::Map<std::string, Gradient>& map_gradient =
			*named_gradients.mutable_name_to_gradient();
		google::protobuf::Map<std::string, Names>& map_names =
			*tuple.mutable_map_names();
		std::vector<std::pair<std::string, tensorflow::Tensor>> feeds;
		std::vector<std::string> actions_to_do;
		actions_to_do.push_back(tuple.training_op_name());
		std::for_each(
			map_gradient.begin(), map_gradient.end(),
			[&feeds, &map_names,
			&tuple](google::protobuf::MapPair<std::string, Gradient>& pair) {
			std::string const& variable_name = pair.first;
			Gradient& grad = pair.second;
			auto iter_map_names = map_names.find(variable_name);
			if (iter_map_names == map_names.end()) {
				std::cout << "this is impossible Line " << __LINE__ << std::endl;
				std::terminate();
			}
			else {
				Names& names = iter_map_names->second;
				std::string grad_name = names.gradient_name();
				std::string index_name = names.gradient_index_name();
				tensorflow::Tensor feed_grad;  // nothing need to do to initialize feed
										  // tensor, dequantize function will do all
										  // stuff
				tensorflow::Tensor feed_index;
				dequantize_gradient(grad, feed_grad);
				feed_index.FromProto(grad.tensor_index());
				std::cout << feed_grad.dim_size(0) << " " << feed_index.dim_size(0) << std::endl;
				feeds.push_back(
					std::pair<std::string, tensorflow::Tensor>(grad_name, feed_grad));
				feeds.push_back(
					std::pair<std::string, tensorflow::Tensor>(index_name, feed_index));
			}
		});
		tensorflow::Status status = sess->Run(feeds, {},  actions_to_do, nullptr);
		if(!status.ok()){
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
		std::cout << "finished update!!!" << std::endl;
	}
	//suitable for state like statistics of gradient
	void moving_average(size_t length, float const * previous, float* current, float const r) {
		for (size_t i = 0; i < length; i++) {
			current[i] = r * previous[i] + (1 - r) * current[i];
		}
	}
	
	float moving_average_v2(float const previous,
		std::vector<float> const& losses,
		std::vector<float> & new_losses, float const r) {
		size_t size = losses.size();
		new_losses.resize(size);
		new_losses[0] = r * previous + (1 - r) * losses[0];
		for (size_t i = 1; i < size; i++) {
			new_losses[i] = r * new_losses[i - 1] + (1 - r) * losses[i];
		}
		return new_losses[size - 1];
	}
	void standard_times(std::vector<float> & times) {
		size_t size = times.size();
		float base = times[0];
		for (int i = 0; i < size; i++) {
			times[i] -= base;
		}
	}

	tensorflow::Tensor get_feed_tensor_from_action(int action_order // begin from 0
	) {
		const size_t total_actions = 3;
		tensorflow::Tensor ret(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({ total_actions }));
		float* ret_ptr = ret.flat<float>().data();
		std::fill(ret_ptr, ret_ptr + total_actions, 0.0f);
		ret_ptr[action_order] = 1.0;
		return ret;
	}

	void add_indices_to_named_gradients(std::map<std::string, tensorflow::Tensor> const & map_indices,
		NamedGradients& named_gradients) {
		for (auto iter = map_indices.begin(); iter != map_indices.end(); iter++) {
			tensorflow::Tensor const & index = iter->second;
			std::string var_name = iter->first;
			auto iter_named_gradients = (named_gradients.mutable_name_to_gradient())->find(var_name);
			Gradient& grad = iter_named_gradients->second;
			index.AsProtoField(grad.mutable_tensor_index());
		}
	}

	namespace {
		bool greater_compare_pair(std::pair<std::string, int> const & a, std::pair<std::string, int> const & b) {
			return b.second < a.second;
		}
	}

	void set_tuple_with_word_to_index(std::string const & material_path, Tuple& tuple) {
		auto & word_to_index = *tuple.mutable_word_to_index();
		std::ifstream input_stream(material_path);
		std::string line;
		while (std::getline(input_stream, line)) {
			std::istringstream iss(line);
			std::string word;
			int index;
			iss >> word;
			iss >> index;
			word_to_index[word] = index;
		}	
	}

	void set_tuple_with_order_to_level(Tuple& tuple) {
		auto & order2level = *tuple.mutable_order_to_level();
		for (int i = 0; i < 5; i++) {
			order2level[i] = 8 + i * 2;
		}
	}

	float get_slope(std::vector<float> const & times, std::vector<float> const & move_average_losses) {
		using namespace Eigen;
		int const size = times.size();
		std::cout << "time is ::" << std::endl;
		for (int i = 0; i < size; i++) {
			std::cout << times[i] << "  ";
		}
		std::cout << std::endl;

		std::cout << "average is ::" << std::endl;
		for (int i = 0; i < size; i++) {
			std::cout << move_average_losses[i] << "  ";
		}
		std::cout << std::endl;
		MatrixXf A = MatrixXf::Random(size, 2);
		VectorXf b = VectorXf::Random(size);
		for (int i = 0; i < size; i++) {
			A(i, 0) = times[i];
			A(i, 1) = 1.0f;
			b(i) = move_average_losses[i];
		}
		//std::cout << A << std::endl << b << std::endl;
		auto qr = A.fullPivHouseholderQr();
		auto w = qr.solve(b);
		std::cout << "slope is " << w << std::endl;
		return w(0);
	}

	void average_gradients(int const number_workers,
		std::map<std::string, tensorflow::Tensor> & name2gradient) {
		auto begin = name2gradient.begin();
		auto end = name2gradient.end();
		for (auto iter = begin; iter != end; iter++) {
			tensorflow::Tensor & tensor = iter->second;
			float* tensor_ptr = tensor.flat<float>().data();
			size_t size = tensor.NumElements();
			std::for_each(tensor_ptr, tensor_ptr + size,
				[number_workers](float& current) { current = current / number_workers; });
		}
	}

	int get_real_level(int const order, int const level) {
		int temp = 0;
		const int min_level = 6;
		const int max_level = 10;
		if (order == 0) {
			temp = level - 1;
			return temp < min_level ? min_level : temp;
		}
		else if (order == 1) {
			return level;
		}
		else if (order == 2) {
			temp = level + 1;
			return temp > max_level ? max_level : temp;
		}
		PRINT_ERROR_MESSAGE("order must between 0 and 2");
		std::terminate();
	}

	int get_real_level_6_8_10(int order) {
		switch (order) {
		case 0:
			return 6;
		case 1:
			return 8;
		case 2:
			return 10;
		}
		PRINT_ERROR_MESSAGE("order should be in the range of 0 to 2");
		std::terminate();
	}
}

