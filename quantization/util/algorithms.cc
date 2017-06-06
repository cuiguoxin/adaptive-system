#include "quantization/util/algorithms.h"
#include "quantization/util/helper.h"

namespace adaptive_system {

	namespace {
		const float eps = 0.000001f;
		void quantize_less_8_bits(
			const QUANTIZATION_TYPE type, float* raw_data, const float max_value,
			const float min_value, const size_t raw_data_length,
			tensorflow::uint8** quantized_data,
			size_t& quantized_data_length  // the number of bytes to output
		) {
			const int q_type = static_cast<int>(type);  // for example 2
			const float scope = std::pow(2, q_type);
			const float multiplizer = scope / (max_value + eps - min_value);
			std::for_each(raw_data, raw_data + raw_data_length,
				[multiplizer, min_value](float& ref) {
				ref = (ref - min_value) * multiplizer;
			});
			const int length_per_iter = 8 / q_type;  // for example 4
			quantized_data_length = static_cast<size_t>(
				std::ceil(raw_data_length / static_cast<float>(length_per_iter)));
			// std::cout << "quantized_data_length is " << quantized_data_length
			//<< std::endl;
			tensorflow::uint8* output = new tensorflow::uint8[quantized_data_length];
			for (int i = 0; i < quantized_data_length; i++) {
				output[i] = 0;
				for (int j = 0; j < length_per_iter; j++) {
					const int index_for_raw = length_per_iter * i + j;
					if (index_for_raw >= raw_data_length) {
						break;
					}
					tensorflow::uint8 value_raw = raw_data[index_for_raw];
					output[i] = output[i] | (value_raw << (q_type * j));
				}
			}
			*quantized_data = output;
		}
		// tensor has been allocated
		void quantize_greater_8_bits(const QUANTIZATION_TYPE type,
			tensorflow::Tensor& raw_data,
			const float max_value, const float min_value,
			tensorflow::Tensor& tensor) {
			float scale_factor = 0.0;
			if (type == QUANTIZATION_TYPE::EIGHT_BIT) {
				scale_factor =
					(static_cast<double>(std::numeric_limits<tensorflow::uint8>::max()) -
						static_cast<double>(std::numeric_limits<tensorflow::uint8>::min())) /
						(max_value + eps - min_value);
				auto o = tensor.flat<tensorflow::uint8>();
				o = ((raw_data.flat<float>().cwiseMin(max_value).cwiseMax(min_value) -
					min_value) *
					scale_factor +
					0.5f)
					.cast<tensorflow::uint8>();
			}
			else if (type == QUANTIZATION_TYPE::SIXTEEN_BIT) {
				scale_factor =
					(static_cast<double>(std::numeric_limits<tensorflow::uint16>::max()) -
						static_cast<double>(std::numeric_limits<tensorflow::uint16>::min())) /
						(max_value + eps - min_value);
				auto o = tensor.flat<tensorflow::uint16>();
				o = ((raw_data.flat<float>().cwiseMin(max_value).cwiseMax(min_value) -
					min_value) *
					scale_factor +
					0.5f)
					.cast<tensorflow::uint16>();
			}
		}

		void dequantize_greater_8_bits(const QUANTIZATION_TYPE type,
			tensorflow::Tensor& quantized_data,
			const float max_range, const float min_range,
			tensorflow::Tensor& tensor) {
			if (type == QUANTIZATION_TYPE::EIGHT_BIT) {
				const float scale_factor =
					(max_range - min_range) /
					(static_cast<float>(std::numeric_limits<tensorflow::uint8>::max()) -
						std::numeric_limits<tensorflow::uint8>::min());

				float* out_ptr = tensor.flat<float>().data();
				const tensorflow::uint8* in_ptr =
					quantized_data.flat<tensorflow::uint8>().data();

				const int num_elements = quantized_data.NumElements();
				for (int i = 0; i < num_elements; ++i) {
					out_ptr[i] = ((static_cast<int>(in_ptr[i])) * scale_factor) + min_range;
				}
			}
			else if (type == QUANTIZATION_TYPE::SIXTEEN_BIT) {
				const float scale_factor =
					(max_range - min_range) /
					(static_cast<float>(std::numeric_limits<tensorflow::uint16>::max()) -
						std::numeric_limits<tensorflow::uint16>::min());

				float* out_ptr = tensor.flat<float>().data();
				const tensorflow::uint16* in_ptr =
					quantized_data.flat<tensorflow::uint16>().data();

				const int num_elements = quantized_data.NumElements();
				for (int i = 0; i < num_elements; ++i) {
					out_ptr[i] = ((static_cast<int>(in_ptr[i])) * scale_factor) + min_range;
				}
			}
		}
		void dequantize_less_8_bits(const QUANTIZATION_TYPE type,
			const tensorflow::uint8* quantized_data,
			const size_t raw_data_length, const float max_value,
			const float min_value, float* raw_data) {
			static const tensorflow::uint8 mask_1_bits = 1,  mask_2_bits = 3, mask_4_bits = 15,
				mask_8_bits = 255;
			const int q_type = static_cast<int>(type);  // for example 2
			// std::cout << "q type is " << q_type << std::endl;
			const int scope = std::pow(2, q_type);
			const int length_per_iter = 8 / q_type;  // for example 4
			const float multiplier = (max_value - min_value) / scope;
			// std::cout << "multiplier is " << multiplier << std::endl;
			int i = 0;
			std::function<void(float&)> func = [=, &i](float& ref) {
				const int index_for_q_data = i / length_per_iter;
				const int index_in_iter = i - index_for_q_data * length_per_iter;
				tensorflow::uint8 q_data = quantized_data[index_for_q_data];
				const int move_right = q_type * index_in_iter;
				q_data = q_data >> move_right;
				switch (q_type) {
				case 1:
					q_data &= mask_1_bits;
					break;
				case 2:
					q_data &= mask_2_bits;
					break;
				case 4:
					q_data &= mask_4_bits;
					break;
				case 8:
					q_data &= mask_8_bits;
					break;
				}
				ref = q_data * multiplier + min_value + multiplier * 0.5;
				++i;
			};
			std::for_each(raw_data, raw_data + raw_data_length, func);
		}
	}
	// raw_tensor ---->>> grad
	void quantize(const QUANTIZATION_TYPE type, tensorflow::Tensor& raw_tensor,
		float const max_value, float const min_value, Gradient& grad) {
		if (type == QUANTIZATION_TYPE::NO_QUANTIZATION) {
			tensorflow::TensorProto* tensor_proto = new tensorflow::TensorProto;
			raw_tensor.AsProtoField(tensor_proto);
			grad.set_allocated_tensor_ge_8(tensor_proto);
			return;
		}
		grad.set_max(max_value);
		grad.set_min(min_value);
		grad.set_level(cast_quantization_type_to_grad_quant_level(type));
		tensorflow::TensorShape raw_tensor_shape = raw_tensor.shape();
		if (type == QUANTIZATION_TYPE::ONE_BIT || type == QUANTIZATION_TYPE::TWO_BIT ||
			type == QUANTIZATION_TYPE::FOUR_BIT) {
			float* raw_ptr = raw_tensor.flat<float>().data();
			int num_elements = raw_tensor.NumElements();
			tensorflow::uint8* out_ptr;
			size_t out_ptr_length = 0;
			quantize_less_8_bits(type, raw_ptr, max_value, min_value, num_elements,
				&out_ptr, out_ptr_length);
			grad.set_tensor_le_8(out_ptr, out_ptr_length);
			delete[] out_ptr;
			tensorflow::TensorShapeProto raw_tensor_shape_proto;
			raw_tensor_shape.AsProto(&raw_tensor_shape_proto);
			*grad.mutable_tensor_shape() = raw_tensor_shape_proto;
		}
		else if (type == QUANTIZATION_TYPE::EIGHT_BIT ||
			type == QUANTIZATION_TYPE::SIXTEEN_BIT) {
			tensorflow::Tensor tensor(cast_quantization_type_to_data_type(type),
				raw_tensor.shape());
			quantize_greater_8_bits(type, raw_tensor, max_value, min_value, tensor);
			tensorflow::TensorProto* tensor_proto = new tensorflow::TensorProto;
			tensor.AsProtoField(tensor_proto);
			grad.set_allocated_tensor_ge_8(tensor_proto);
		}
	}

	// grad --->>> raw_tensor
	// raw_tensor can be an empty tensor
	void dequantize(const QUANTIZATION_TYPE type, Gradient& grad,
		tensorflow::Tensor& raw_tensor) {
		if (type == QUANTIZATION_TYPE::NO_QUANTIZATION) {
			const tensorflow::TensorProto& tensor_proto = grad.tensor_ge_8();
			bool success = raw_tensor.FromProto(tensor_proto);
			if (!success) {
				PRINT_ERROR_MESSAGE("tensorflow::tensor::fromProto failed");
				std::terminate();
			}
			return;
		}
		float const max_value = grad.max();
		float const min_value = grad.min();
		if (type == QUANTIZATION_TYPE::ONE_BIT || type == QUANTIZATION_TYPE::TWO_BIT ||
			type == QUANTIZATION_TYPE::FOUR_BIT) {
			tensorflow::TensorShapeProto* tensor_shape_proto =
				grad.mutable_tensor_shape();
			tensorflow::TensorShape tensor_shape(*tensor_shape_proto);
			raw_tensor =
				tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, tensor_shape);
			float* raw_ptr = raw_tensor.flat<float>().data();
			int number_raw = raw_tensor.NumElements();

			tensorflow::uint8 const* quantized_ptr =
				reinterpret_cast<const unsigned char*>(
					grad.mutable_tensor_le_8()->data());
			dequantize_less_8_bits(type, quantized_ptr, number_raw, max_value,
				min_value, raw_ptr);
		}
		else if (type == QUANTIZATION_TYPE::EIGHT_BIT ||
			type == QUANTIZATION_TYPE::SIXTEEN_BIT) {
			tensorflow::TensorProto& tensor_quantized_proto =
				*grad.mutable_tensor_ge_8();
			tensorflow::Tensor temp;
			bool is_ok = temp.FromProto(tensor_quantized_proto);
			if (!is_ok) {
				std::cout << "proto to tensor failed in line " << __LINE__ << std::endl;
				std::terminate();
			}
			// must allocate tensor memory outside dequantize_greater_8_bits
			raw_tensor =
				tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, temp.shape());
			dequantize_greater_8_bits(type, temp, max_value, min_value, raw_tensor);
		}
	}

	void get_max_and_min_value(tensorflow::Tensor const& tensor, float& max,
		float& min) {
		float const* tensor_ptr = tensor.flat<float>().data();
		size_t size = tensor.NumElements();
		max = tensor_ptr[0];
		min = tensor_ptr[0];
		std::for_each(tensor_ptr, tensor_ptr + size,
			[&max, &min](float const current) {
			if (max < current) {
				max = current;
				return;
			}
			if (min > current) {
				min = current;
			}
		});
	}
	// map_gradient ---->>>named_gradients
	void quantize_gradient(std::map<std::string, tensorflow::Tensor>& map_gradient,
		NamedGradients* named_gradients,
		QUANTIZATION_TYPE type) {
		std::for_each(map_gradient.begin(), map_gradient.end(),
			[named_gradients,
			type](std::pair<std::string const, tensorflow::Tensor>& pair) {
			std::string const& variable_name = pair.first;
			tensorflow::Tensor& raw_tensor = pair.second;
			float max = 0, min = 0;
			get_max_and_min_value(raw_tensor, max, min);
			Gradient grad;
			quantize(type, raw_tensor, max, min, grad);
			named_gradients->mutable_name_to_gradient()->insert(
				google::protobuf::MapPair<::std::string, Gradient>(
					variable_name, grad));
		});
	}

	// put named quantized gradients into a dequantized tensor map
	void dequantize_gradient(
		NamedGradients& named_gradients,
		std::map<std::string, tensorflow::Tensor>& map_gradient) {
		auto map_quantized_gradient =
			named_gradients.mutable_name_to_gradient();  // const reference
		std::for_each(
			map_quantized_gradient->begin(), map_quantized_gradient->end(),
			[&map_gradient](
				::google::protobuf::MapPair<std::string, Gradient>& pair) {
			Gradient& gradient = pair.second;
			std::string const& variable_name = pair.first;
			tensorflow::Tensor temp_tensor;
			dequantize(cast_grad_quant_level_to_quantization_type(gradient.level()),
				gradient, temp_tensor);
			map_gradient.insert(std::pair<std::string, tensorflow::Tensor>(
				variable_name, temp_tensor));
		});
	}

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
			[&feeds, &actions_to_do, &map_names,
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
				std::string assign_add_name = names.assign_add_name();

				tensorflow::Tensor feed;  // nothing need to do to initialize feed
										  // tensor, dequantize function will do all
										  // stuff
				dequantize(cast_grad_quant_level_to_quantization_type(grad.level()),
					grad, feed);
				float* feed_ptr = feed.flat<float>().data();
				float learning_rate = tuple.lr();
				std::for_each(
					feed_ptr, feed_ptr + feed.NumElements(),
					[&learning_rate](float& ref) { ref = -ref * learning_rate; });
				feeds.push_back(std::pair<std::string, tensorflow::Tensor>(
					names.placeholder_assign_add_name(), feed));
				actions_to_do.push_back(assign_add_name);
			}
		});
		sess->Run(feeds, {}, actions_to_do, nullptr);
	}

	void apply_quantized_gradient_to_model_using_adam(
		NamedGradients& named_gradients, tensorflow::Session* sess, Tuple& tuple) {
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

				tensorflow::Tensor feed;  // nothing need to do to initialize feed
										  // tensor, dequantize function will do all
										  // stuff
				dequantize(cast_grad_quant_level_to_quantization_type(grad.level()),
					grad, feed);
				feeds.push_back(
					std::pair<std::string, tensorflow::Tensor>(grad_name, feed));
			}
		});
		sess->Run(feeds, {}, actions_to_do, nullptr);
	}

	GRAD_QUANT_LEVEL cast_quantization_type_to_grad_quant_level(
		QUANTIZATION_TYPE type) {
		switch (type) {
		case QUANTIZATION_TYPE::ONE_BIT:
			return GRAD_QUANT_LEVEL::ONE;
		case QUANTIZATION_TYPE::TWO_BIT:
			return GRAD_QUANT_LEVEL::TWO;
		case QUANTIZATION_TYPE::FOUR_BIT:
			return GRAD_QUANT_LEVEL::FOUR;
		case QUANTIZATION_TYPE::EIGHT_BIT:
			return GRAD_QUANT_LEVEL::EIGHT;
		case QUANTIZATION_TYPE::SIXTEEN_BIT:
			return GRAD_QUANT_LEVEL::SIXTEEN;
		}
		return GRAD_QUANT_LEVEL::NONE;
	}
	QUANTIZATION_TYPE cast_grad_quant_level_to_quantization_type(
		GRAD_QUANT_LEVEL const level) {
		switch (level) {
		case GRAD_QUANT_LEVEL::ONE:
			return QUANTIZATION_TYPE::ONE_BIT;
		case GRAD_QUANT_LEVEL::TWO:
			return QUANTIZATION_TYPE::TWO_BIT;
		case GRAD_QUANT_LEVEL::FOUR:
			return QUANTIZATION_TYPE::FOUR_BIT;
		case GRAD_QUANT_LEVEL::EIGHT:
			return QUANTIZATION_TYPE::EIGHT_BIT;
		case GRAD_QUANT_LEVEL::SIXTEEN:
			return QUANTIZATION_TYPE::SIXTEEN_BIT;
		}
		//std::terminate();
		return QUANTIZATION_TYPE::NO_QUANTIZATION;
	}
	tensorflow::DataType cast_quantization_type_to_data_type(
		QUANTIZATION_TYPE type) {
		if (type == QUANTIZATION_TYPE::EIGHT_BIT) {
			return tensorflow::DataType::DT_UINT8;
		}
		else if (type == QUANTIZATION_TYPE::SIXTEEN_BIT) {
			return tensorflow::DataType::DT_UINT16;
		}
		std::terminate();
		return tensorflow::DataType::DT_INVALID;
	}
}
