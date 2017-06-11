#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include "proto/rpc_service.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#ifndef ADAPTIVE_SYSTEM_ALGORITHM_H
#define ADAPTIVE_SYSTEM_ALGORITHM_H
namespace adaptive_system {

	enum QUANTIZATION_TYPE {
		ONE_BIT = 1,
		TWO_BIT = 2,
		FOUR_BIT = 4,
		EIGHT_BIT = 8,
		SIXTEEN_BIT = 16,
		NO_QUANTIZATION = 32
	};
	void quantize(const QUANTIZATION_TYPE type,
		tensorflow::Tensor const& raw_tensor, float const max_value,
		float const min_value, Gradient& grad);
	void dequantize(const QUANTIZATION_TYPE type, Gradient& grad,
		tensorflow::Tensor& raw_tensor);

	GRAD_QUANT_LEVEL cast_quantization_type_to_grad_quant_level(
		QUANTIZATION_TYPE type);
	QUANTIZATION_TYPE cast_grad_quant_level_to_quantization_type(
		GRAD_QUANT_LEVEL const level);
	tensorflow::DataType cast_quantization_type_to_data_type(
		QUANTIZATION_TYPE type);

	void get_max_and_min_value(tensorflow::Tensor const& tensor, float& max,
		float& min);

	void quantize_gradient(std::map<std::string, tensorflow::Tensor>& map_gradient,
		NamedGradients* named_gradients, QUANTIZATION_TYPE type);

	void dequantize_gradient(
		NamedGradients& named_gradients,
		std::map<std::string, tensorflow::Tensor>& map_gradient);

	void apply_quantized_gradient_to_model(NamedGradients& named_gradients,
		tensorflow::Session* sess, Tuple& tuple);


	void moving_average(size_t length, float const * previous, float* current, const float r);
}
#endif
