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

	void apply_quantized_gradient_to_model(NamedGradientsAccordingColumn& named_gradients,
		tensorflow::Session* sess, Tuple& tuple);


	void moving_average(size_t length, float const * previous, float* current, const float r);

	float moving_average_v2(float const previous,
		std::vector<float> const& losses,
		std::vector<float> & new_losses, float const r);

	void moving_average_then_minus_average(
		std::vector<float> const& losses,
		std::vector<float> & new_losses, float const r);

	void moving_average_not_minus_average(
		std::vector<float> const& losses,
		std::vector<float> & new_losses, float const r);

	float moving_average_from_last_loss(float const last_loss,
		std::vector<float> const& losses,
		std::vector<float> & new_losses,
		float const r);

	float minus_average_then_moving_average(
		std::vector<float> const& losses,
		std::vector<float> & new_losses, float const r);

	void standard_times(std::vector<float> & times);

	tensorflow::Tensor get_feed_tensor_from_action(int action_order);

	float get_slope(std::vector<float> const & times, std::vector<float> const & move_average_losses);

	void average_gradients(int const number_workers, std::map<std::string, tensorflow::Tensor> & name2gradient);

	int get_real_level(int const order, int const level);

	int get_real_level_6_8_10(int order);

	void aggregate_gradients(std::vector<std::map<std::string, tensorflow::Tensor>>& vector_of_map, 
		std::map<std::string, tensorflow::Tensor> & return_result);

	tensorflow::Tensor get_float_tensor_from_vector(const std::vector<float> & vec);

	float get_slope_according_loss(const std::vector<float> & loss_vec);
}
#endif
