#ifndef ACTOR_CRITIC_H
#define ACTOR_CRITIC_H
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include "proto/rpc_service.grpc.pb.h"
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

#include "quantization/util/algorithms.h"
#include "quantization/util/extract_feature.h"
#include "quantization/util/helper.h"

namespace adaptive_system {
	class actor_critic {
	private:
		tensorflow::Session* _session;
		std::string _sarsa_model_path;
		float _r;
		float _beta;
		size_t _T;
	public:
		actor_critic(std::string const & model_path, float const r, float const beta, size_t t);
		GRAD_QUANT_LEVEL sample_action_from_policy(tensorflow::Tensor const & state);
		float get_update_value(float reward, tensorflow::Tensor const & new_state, 
			tensorflow::Tensor const & last_state);
		void update_value_function_parameter(tensorflow::Tensor const & state, const float update);
		void update_policy_parameter(tensorflow::Tensor const & state, const float update);

	};
}
#endif // !ACTOR_CRITIC_H
