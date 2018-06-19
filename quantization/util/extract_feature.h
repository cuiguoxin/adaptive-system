#ifndef EXTRACT_FEATURE_H
#define EXTRACT_FEATURE_H


#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <grpc++/grpc++.h>

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

namespace adaptive_system {
	tensorflow::Tensor get_feature(tensorflow::Tensor const& tensor, const float loss);
	tensorflow::Tensor get_feature_v2(tensorflow::Tensor const & tensor,
		std::vector<float> const & recent_losses);
	tensorflow::Tensor get_final_state_from_partial_state(std::vector<PartialState> const & vector_partial_states);
}

#endif // !EXTRACT_FEATURE_H

