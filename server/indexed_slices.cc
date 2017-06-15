#include "server/indexed_slices.h"
#include "quantization/util/helper.h"
#include "proto/rpc_service.pb.h"
#include <functional>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace tensorflow;

namespace adaptive_system {

	namespace {
		size_t get_embedding_dimension(std::pair<Tensor, Tensor>const & p) {
			Tensor const & values = p.first;
			if (values.shape().dims() == 1) {
				return 1;
			}
			return values.shape().dim_size(1);
		}

		std::vector<Tensor> split_tensor_according_to_first_dimension(Tensor const & tensor) {
			std::vector<Tensor> ret_vec;
			size_t first_dimension_size = tensor.shape().dim_size(0);
			size_t second_dimension_size = 0;
			if (tensor.shape().dims() == 1) {
				second_dimension_size = 1;
			}
			else {
				second_dimension_size = tensor.shape().dim_size(1);
			}
			float const* value_ptr = tensor.flat<float>().data();
			size_t current_index = 0;
			for (int i = 0; i < first_dimension_size; i++) {
				Tensor temp(DataType::DT_FLOAT, TensorShape({ second_dimension_size }));
				float * temp_ptr = temp.flat<float>().data();
				std::copy(value_ptr + current_index,
					value_ptr + current_index + second_dimension_size,
					temp_ptr);
				current_index = current_index + second_dimension_size;
				ret_vec.push_back(temp);
			}
			return ret_vec;
		}

		void push_or_add(int const index, const Tensor & value,
			std::unordered_map<int, Tensor>& index_to_value) {
			auto iter = index_to_value.find(index);
			if (iter == index_to_value.end()) {
				index_to_value[index] = value;
			}
			else {
				float* ptr = iter->second.flat<float>().data();
				float const* ptr_value = value.flat<float>().data();
				//add ptr_value to ptr
				size_t size = value.NumElements();
				for (int i = 0; i < size; i++) {
					ptr[i] += ptr_value[i];
				}
			}
		}
	}
	//pair<values, indices>
	std::pair<Tensor, Tensor>  merge_multiple_indexed_slices
	(std::vector<std::pair<Tensor, Tensor>> const & vec_index_slice) {
		std::unordered_map<int, Tensor> index_to_value;
		const size_t size = vec_index_slice.size();
		const size_t embedding_dimension = get_embedding_dimension(vec_index_slice[0]);
		//merge into index_to_value
		DataType type;
		for (int i = 0; i < size; i++) {
			const std::pair<Tensor, Tensor> current_pair = vec_index_slice[i];
			const Tensor& value = current_pair.first;
			const Tensor& index = current_pair.second;
			std::vector<Tensor> tensors_value = split_tensor_according_to_first_dimension(value);
			const size_t numbers = index.shape().dim_size(0);
			DataType d_type = index.dtype();
			if (d_type == DataType::DT_INT32) {
				int32_t* index_ptr = index.flat<int32_t>().data();
				for (int j = 0; j < numbers; j++) {
					push_or_add(index_ptr[j], tensors_value[j], index_to_value);
				}
				type = DataType::DT_INT32;
			}
			else if (d_type = DataType::DT_INT64) {
				int64_t* index_ptr = index.flat<int64_t>().data();
				for (int j = 0; j < numbers; j++) {
					push_or_add(index_ptr[j], tensors_value[j], index_to_value);
				}
				type = DataType::DT_INT64;
			}
			else {
				PRINT_ERROR_MESSAGE("dtype is not right");
				std::terminate();
			}
		}
		//put index_to_value into ret
		size_t const map_size = index_to_value.size();
		TensorShape shape;
		if (embedding_dimension == 1) {
			shape = TensorShape({ map_size });
		}
		else {
			shape = TensorShape({ map_size, embedding_dimension });
		}
		Tensor ret_value(DataType::DT_FLOAT, shape);
		float* ret_value_ptr = ret_value.flat<float>().data();
		Tensor ret_index(type, TensorShape({ map_size }));
		if (type == DataType::DT_INT32) {
			int32_t* ret_index_ptr = ret_index.flat<int32_t>().data();
			size_t current_index_value = 0;
			size_t current_index_index = 0;
			for (auto iter = index_to_value.begin(); iter != index_to_value.end(); iter++) {
				int index = iter->first;
				Tensor const & value_tensor = iter->second;
				float const * value_tensor_ptr = value_tensor.flat<float>().data();
				ret_index_ptr[current_index_index++] = index;
				std::copy(value_tensor_ptr,
					value_tensor_ptr + embedding_dimension, ret_value_ptr + current_index_value);
				current_index_value += embedding_dimension;
			}
		}
		else {
			int64_t* ret_index_ptr = ret_index.flat<int64_t>().data();
			size_t current_index_value = 0;
			size_t current_index_index = 0;
			for (auto iter = index_to_value.begin(); iter != index_to_value.end(); iter++) {
				int index = iter->first;
				Tensor const & value_tensor = iter->second;
				float const * value_tensor_ptr = value_tensor.flat<float>().data();
				ret_index_ptr[current_index_index++] = index;
				std::copy(value_tensor_ptr,
					value_tensor_ptr + embedding_dimension, ret_value_ptr + current_index_value);
				current_index_value += embedding_dimension;
			}
		}
		return std::make_pair(ret_value, ret_index);
	}

	void extract_indices_from_named_gradient(const NamedGradients& named_gradients,
		std::map<std::string, Tensor>& map_indices) {
		auto& map_named_gradients = named_gradients.name_to_gradient();
		for (auto iter = map_named_gradients.begin(); iter != map_named_gradients.end(); iter++) {
			std::string var_name = iter->first;
			Tensor index;
			index.FromProto(iter->second.tensor_index());
			map_indices.insert(std::pair<std::string, Tensor>(var_name, index));
		}
	}

	void aggregate_indexed_slices(std::vector<std::map<std::string, Tensor>> const & vec_map_gradients,
		std::vector<std::map<std::string, Tensor>> const & vec_map_indices,
		std::map<std::string, Tensor>& merged_gradients,
		std::map<std::string, Tensor>& merged_indices) {
		merged_gradients.clear();
		merged_indices.clear();
		size_t size = vec_map_indices.size();
		//var_name --->>> vector<pair<grad, indice>>
		std::map<std::string, std::vector<std::pair<Tensor, Tensor>>> name_to_vec_pair;
		for (int i = 0; i < size; i++) {
			auto& map_indices = vec_map_indices[i];
			auto& map_gradients = vec_map_gradients[i];
			for (auto iter = map_indices.begin(); iter != map_indices.end(); iter++) {
				std::string var_name = iter->first;
				auto & indice_tensor = iter->second;
				auto & gradient_tensor = map_gradients.find(var_name)->second;
				name_to_vec_pair[var_name].push_back(std::pair<Tensor, Tensor>(gradient_tensor, indice_tensor));
			}
		}
		for (auto iter = name_to_vec_pair.begin(); iter != name_to_vec_pair.end(); iter++) {
			std::string var_name = iter->first;
			auto& vec_pair = iter->second;
			auto merged_pair = merge_multiple_indexed_slices(vec_pair);
			merged_gradients[var_name] = merged_pair.first;
			merged_indices[var_name] = merged_pair.second;
		}

	}
}