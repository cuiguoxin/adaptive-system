#include "quantization/util/extract_feature.h"
#include "quantization/util/helper.h"
namespace adaptive_system {
	namespace {

		void mean(tensorflow::Tensor const & tensor, float& result) {
			size_t size = tensor.NumElements();
			float const* tensor_ptr = tensor.flat<float>().data();
			float sum = 0;
			for (size_t i = 0; i < size; i++) {
				sum += tensor_ptr[i];
			}
			result = sum / size;
		}
		void max(tensorflow::Tensor const & tensor, float& result) {
			size_t size = tensor.NumElements();
			float const* tensor_ptr = tensor.flat<float>().data();
			result = tensor_ptr[0];
			for (size_t i = 1; i < size; i++) {
				if (result < tensor_ptr[i]) result = tensor_ptr[i];
			}
		}
		void min(tensorflow::Tensor const &tensor, float& result) {
			size_t size = tensor.NumElements();
			float const* tensor_ptr = tensor.flat<float>().data();
			result = tensor_ptr[0];
			for (size_t i = 1; i < size; i++) {
				if (result > tensor_ptr[i]) result = tensor_ptr[i];
			}
		}
		void deviation(tensorflow::Tensor const& tensor, float& result) {
			size_t size = tensor.NumElements();
			float const* tensor_ptr = tensor.flat<float>().data();
			float sum = 0;
			for (size_t i = 0; i < size; i++) {
				sum += tensor_ptr[i];
			}
			float average = sum / size;
			float deviation_sum = 0;
			for (size_t i = 0; i < size; i++) {
				deviation_sum += std::pow(tensor_ptr[i] - average, 2.0);
			}
			deviation_sum = deviation_sum / size;
			result = std::sqrt(deviation_sum);
		}
		void abs_sum(tensorflow::Tensor const& tensor, float& result) {
			size_t size = tensor.NumElements();
			float const* tensor_ptr = tensor.flat<float>().data();
			result = 0;
			for (size_t i = 0; i < size; i++) {
				if (tensor_ptr[i] > 0)
					result += tensor_ptr[i];
				else
					result -= tensor_ptr[i];
			}
			result = result / 1000;
		}
		void median(tensorflow::Tensor const& tensor, float& result) {
			size_t size = tensor.NumElements();
			float const* tensor_ptr = tensor.flat<float>().data();
			float* float_new = new float[size];
			std::copy(tensor_ptr, tensor_ptr + size, float_new);
			std::nth_element(float_new, float_new + size / 2, float_new + size);
			result = float_new[size / 2];
			delete[] float_new;
		}
		void norm(tensorflow::Tensor const& tensor, float& result) {
			size_t size = tensor.NumElements();
			float const* tensor_ptr = tensor.flat<float>().data();
			result = 0;
			for (size_t i = 0; i < size; i++) {
				result += std::pow(tensor_ptr[i], 2.0);
			}
			result = std::sqrt(result) / 10;
		}
	}

	tensorflow::Tensor get_feature(tensorflow::Tensor const& tensor, const float loss) {
		tensorflow::Tensor ret_tensor =
			tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({ 8 }));
		float* ret_tensor_ptr = ret_tensor.flat<float>().data();
		std::thread mean_thread(mean, std::ref(tensor), std::ref(ret_tensor_ptr[0]));
		std::thread min_thread(min, std::ref(tensor), std::ref(ret_tensor_ptr[1]));
		std::thread max_thread(max, std::ref(tensor), std::ref(ret_tensor_ptr[2]));
		std::thread deviation_thread(deviation, std::ref(tensor),
			std::ref(ret_tensor_ptr[3]));
		std::thread abs_sum_thread(abs_sum, std::ref(tensor),
			std::ref(ret_tensor_ptr[4]));
		std::thread median_thread(median, std::ref(tensor),
			std::ref(ret_tensor_ptr[5]));
		std::thread norm_thread(norm, std::ref(tensor), std::ref(ret_tensor_ptr[6]));
		mean_thread.join();
		min_thread.join();
		max_thread.join();
		deviation_thread.join();
		abs_sum_thread.join();
		median_thread.join();
		norm_thread.join();
		ret_tensor_ptr[7] = loss;

		return ret_tensor;
	}
	//only contains deviation, abs_sum, norm and recent losses
	tensorflow::Tensor get_feature_v2(tensorflow::Tensor const & tensor,
		std::vector<float> const & recent_losses) {
		size_t const recent_loss_size = recent_losses.size();
		size_t const statistic_size
		tensorflow::Tensor ret_tensor =
			tensorflow::Tensor(tensorflow::DataType::DT_FLOAT,
				tensorflow::TensorShape({ 3 + recent_loss_size}));
		float* ret_tensor_ptr = ret_tensor.flat<float>().data();
		std::thread deviation_thread(deviation, std::ref(tensor),
			std::ref(ret_tensor_ptr[0]));
		std::thread abs_sum_thread(abs_sum, std::ref(tensor),
			std::ref(ret_tensor_ptr[1]));
		std::thread norm_thread(norm, std::ref(tensor),
			std::ref(ret_tensor_ptr[2]));
		
		deviation_thread.join();
		abs_sum_thread.join();
		norm_thread.join();

		for (int i = 0; i < recent_loss_size; i++) {
			ret_tensor_ptr[i + statistic_size] = recent_losses[i];
		}
		

		return ret_tensor;
	}

	tensorflow::Tensor get_final_state_from_partial_state(std::vector<PartialState>const & vector_partial_states) {
		size_t vector_size = vector_partial_states.size();
		const size_t state_length = 8;
		tensorflow::Tensor tensor_ret(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({ state_length }));
		float* tensor_ret_ptr = tensor_ret.flat<float>().data();
		std::fill(tensor_ret_ptr, tensor_ret_ptr + state_length, 0.0f);
		for (int i = 0; i < vector_size; i++) {
			tensorflow::Tensor current_partial_tensor;
			bool is_success = current_partial_tensor.FromProto(vector_partial_states[i].tensor());
			if(!is_success){
				PRINT_ERROR_MESSAGE("FROM PROTO FAILED");
				std::terminate();
			}
			float* current_partial_tensor_ptr = current_partial_tensor.flat<float>().data();
			for (int j = 0; j < state_length; j++) {
				tensor_ret_ptr[j] += current_partial_tensor_ptr[j];
			}
		}
		std::for_each(tensor_ret_ptr, tensor_ret_ptr + state_length, [vector_size](float& ref) {
			ref = ref / vector_size;
		});
		return tensor_ret;
	}
}
