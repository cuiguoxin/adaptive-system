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
#include <thread>

namespace adaptive_system {

	void apply_quantized_gradient_to_model(NamedGradientsAccordingColumn& named_gradients,
		tensorflow::Session* sess,
		Tuple& tuple) {
		google::protobuf::Map<std::string, GradientAccordingColumn>& map_gradient =
			*named_gradients.mutable_name_to_gradient();
		google::protobuf::Map<std::string, Names>& map_names =
			*tuple.mutable_map_names();
		std::vector<std::pair<std::string, tensorflow::Tensor>> feeds;
		std::vector<std::string> actions_to_do;
		actions_to_do.push_back(tuple.training_op_name());
		std::for_each(
			map_gradient.begin(), map_gradient.end(),
			[&feeds, &map_names,
			&tuple](google::protobuf::MapPair<std::string, GradientAccordingColumn>& pair) {
			std::string const& variable_name = pair.first;
			GradientAccordingColumn& grad = pair.second;
			auto iter_map_names = map_names.find(variable_name);
			if (iter_map_names == map_names.end()) {
				std::cout << "this is impossible Line " << __LINE__ << std::endl;
				std::terminate();
			}
			else {
				Names& names = iter_map_names->second;
				std::string grad_name = names.gradient_name();
				tensorflow::Tensor feed_grad;  // nothing need to do to initialize feed
										  // tensor, dequantize function will do all
										  // stuff
				bool is_quantized = grad.is_quantized();
				if (is_quantized) {
					dequantize_gradient_according_column(grad, feed_grad);
				}
				else {
					feed_grad.FromProto(grad.tensor());
				}
				
				feeds.push_back(
					std::pair<std::string, tensorflow::Tensor>(grad_name, feed_grad));
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

	

	namespace {
		bool greater_compare_pair(std::pair<std::string, int> const & a, std::pair<std::string, int> const & b) {
			return b.second < a.second;
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

	namespace {
		void sum_tensor_vector(std::vector<tensorflow::Tensor> const & vec_tensor,
			tensorflow::Tensor& out_tensor) {
			auto shape = vec_tensor[0].shape();
			tensorflow::Tensor return_tensor(tensorflow::DataType::DT_FLOAT, shape);
			size_t size = return_tensor.NumElements();
			std::cout << "size is " << size << std::endl;
			float* return_tensor_ptr = return_tensor.flat<float>().data();
			std::fill(return_tensor_ptr, return_tensor_ptr + size, 0.0f);
			for (int i = 0; i < vec_tensor.size(); i++) {
				auto & tensor = vec_tensor[i];
				float const * tensor_ptr = tensor.flat<float>().data();
				for (int j = 0; j < size; j++) {
					return_tensor_ptr[j] += tensor_ptr[j];
				}
			}
			out_tensor = std::move(return_tensor);
		}
	}

	void aggregate_gradients(std::vector<std::map<std::string, tensorflow::Tensor>>& vector_of_map,
		std::map<std::string, tensorflow::Tensor> & return_result) {
		std::map<std::string, std::vector<tensorflow::Tensor>> map_tensor_vector;
		PRINT_INFO;
		for (auto iter = vector_of_map.begin(); iter != vector_of_map.end(); iter++) {
			std::map<std::string, tensorflow::Tensor> & map_current = *iter;
			for (auto iter_name_tensor = map_current.begin();
				iter_name_tensor != map_current.end(); iter_name_tensor++) {
				std::string var_name = iter_name_tensor->first;
				tensorflow::Tensor& gradient = iter_name_tensor->second;
				map_tensor_vector[var_name].push_back(gradient);
			}
		}
		PRINT_INFO;
		std::vector<std::thread> vector_threads;
		std::vector<std::pair<std::string, tensorflow::Tensor>> vector_name_tensor;
		vector_name_tensor.resize(map_tensor_vector.size());
		int index = 0; 
		for (auto iter = map_tensor_vector.begin(); iter != map_tensor_vector.end(); iter++) {
			std::string var_name = iter->first;
			auto& vector_tensor = iter->second;
			vector_name_tensor[index].first = var_name;
			auto& ref_tensor = vector_name_tensor[index++].second;
			vector_threads.push_back(
				std::thread(sum_tensor_vector, std::ref(vector_tensor), std::ref(ref_tensor)));
		}
		PRINT_INFO;
		for (auto iter = vector_threads.begin(); iter != vector_threads.end(); iter++) {
			iter->join();
		}
		PRINT_INFO;
		return_result.clear();
		for (auto iter = vector_name_tensor.begin(); iter != vector_name_tensor.end(); iter++) {
			return_result[iter->first] = iter->second;
		}
		PRINT_INFO;
	}

}

