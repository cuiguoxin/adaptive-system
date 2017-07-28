#ifndef ADAPTIVE_SYSTEM_ALGORITHM_H
#define ADAPTIVE_SYSTEM_ALGORITHM_H

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

#include "quantization/util/any_level.h"
#include <Eigen/Dense>


namespace adaptive_system {

	void apply_quantized_gradient_to_model(NamedGradients& named_gradients,
		tensorflow::Session* sess, Tuple& tuple);


	void moving_average(size_t length, float const * previous, float* current, const float r);

	float moving_average_v2(float const previous,
		std::vector<float> const& losses,
		std::vector<float> & new_losses, float const r);

	void standard_times(std::vector<float> & times);

	tensorflow::Tensor get_feed_tensor_from_action(int action_order);

	void add_indices_to_named_gradients(std::map<std::string, tensorflow::Tensor> const & map_indices,
		NamedGradients& named_gradients);

	void set_tuple_with_word_to_index(std::string const & material_path, Tuple& tuple);

	void set_tuple_with_order_to_level(Tuple& tuple);

	float get_slope(std::vector<float> const & times, std::vector<float> const & move_average_losses);

	void average_gradients(int const number_workers, std::map<std::string, tensorflow::Tensor> & name2gradient);

	int get_real_level(int const order, int const level);

	int get_real_level_6_8_10(int order);
}
#endif
