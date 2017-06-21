#include "quantization/util/algorithms.h"
#include "quantization/util/helper.h"

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
		
		std::for_each(
			map_gradient.begin(), map_gradient.end(),
			[&feeds, &map_names,
			&tuple, &actions_to_do](google::protobuf::MapPair<std::string, Gradient>& pair) {
			std::string const& variable_name = pair.first;
			Gradient& grad = pair.second;
			auto iter_map_names = map_names.find(variable_name);
			if (iter_map_names == map_names.end()) {
				std::cout << "this is impossible Line " << __LINE__ << std::endl;
				std::terminate();
			}
			else {
				Names& names = iter_map_names->second;
				std::string grad_name = names.placeholder_gradient_name();
				std::string index_name = names.placeholder_indice_name();
				std::string scatter_sub_name = names.scatter_sub_name();
				actions_to_do.push_back(scatter_sub_name);

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

	void moving_average(size_t length, float const * previous, float* current, float const r) {
		for (size_t i = 0; i < length; i++) {
			current[i] = r * previous[i] + (1 - r) * current[i];
		}
	}
	
	tensorflow::Tensor get_feed_tensor_from_action(int action_order // begin from 0
	) {
		const size_t total_actions = 5;
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
}

