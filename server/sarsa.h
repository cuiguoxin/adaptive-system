#ifndef SARSA_H
#define SARSA_H

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
	class sarsa_model {
	private:
		tensorflow::Session* _session;
		std::string _sarsa_model_path;
		float _r;
		float _eps_greedy;
		
		std::vector<float> get_greedy_probability(size_t index_of_max);

	public:
		sarsa_model(std::string const& path, float r, float eps_greedy);

		float get_q_value(tensorflow::Tensor const& state, GRAD_QUANT_LEVEL action);

		GRAD_QUANT_LEVEL sample_new_action(tensorflow::Tensor const& state);

		void adjust_model(float reward, tensorflow::Tensor const& old_state,
			GRAD_QUANT_LEVEL old_action,
			tensorflow::Tensor const& new_state,
			GRAD_QUANT_LEVEL new_action);

	};
}


#endif // !SARSA_H

