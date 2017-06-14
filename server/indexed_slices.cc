#include "server/indexed_slices.h"
#include <functional>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace tensorflow;

namespace adaptive_system {

	namespace {
		size_t get_embedding_dimension(std::pair<Tensor, Tensor>const & p) {
			Tensor const & values = p.fist;
			return values.shape().dim_size(1);
		}

		std::vector<Tensor> split_tensor_according_to_first_dimension(Tensor const & tensor) {
			std::vector<Tensor> ret_vec;
			size_t first_dimension_size = tensor.shape().dim_size(0);
			size_t second_dimension_size = tensor.shape().dim_size(1);
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
	}
	//pair<values, indices>
	std::pair<Tensor, Tensor>  merge_multiple_indexed_slices
	(std::vector<std::pair<Tensor, Tensor>> const & vec_index_slice) {
		std::unordered_map<int, Tensor> index_to_value;
		const size_t size = vec_index_slice.size();
		const size_t embedding_dimension = get_embedding_dimension(vec_index_slice[0]);
		for (int i = 0; i < size; i++) {
			const std::pair<Tensor, Tensor> current_pair = vec_index_slice[i];
			const Tensor& value = current_pair.first;
			const Tensor& index = current_pair.second;
			std::vector<Tensor> tensors_value = split_tensor_according_to_first_dimension(value);

		}
	}
}