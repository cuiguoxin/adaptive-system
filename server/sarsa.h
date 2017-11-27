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
		const int _start_level;
		const int _end_level;
		int _current_level;
		float _r;
		float _eps_greedy;
		//std::vector<float> _probability;
		std::discrete_distribution<int> _discrete;
		
		//std::vector<float> get_greedy_probability(size_t index_of_max);
		int get_index_from_level(int level);
		int get_level_from_index(int index);
		int get_total_level_number();
		tensorflow::Tensor get_feed_tensor_from_index(int index);
		int get_current_level();
		int get_current_index();
		int index_of_max(float* array);
		int sample();

	public:
		sarsa_model(std::string const& path, int const input_size, float r, float eps_greedy, int start, int end, int init);

		float get_q_value(tensorflow::Tensor const& state, int action_order);

		int sample_new_action(tensorflow::Tensor const& state);

		void adjust_model(float reward, tensorflow::Tensor const& old_state,
			int const old_action_order,
			tensorflow::Tensor const& new_state,
			int const new_action_order);

	};
}


#endif // !SARSA_H

