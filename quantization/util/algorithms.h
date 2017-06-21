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


namespace adaptive_system {

	void apply_quantized_gradient_to_model(NamedGradients& named_gradients,
		tensorflow::Session* sess, Tuple& tuple);


	void moving_average(size_t length, float const * previous, float* current, const float r);

	tensorflow::Tensor get_feed_tensor_from_action(int action_order);

	void add_indices_to_named_gradients(std::map<std::string, tensorflow::Tensor> const & map_indices,
		NamedGradients& named_gradients);

	void set_tuple_with_word_to_index(std::string const & material_path, Tuple& tuple);
}
#endif
