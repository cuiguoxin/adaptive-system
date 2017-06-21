#ifndef REWARD_H
#define REWARD_H


#include <iostream>
#include <algorithm>
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

#include "proto/rpc_service.pb.h"

namespace adaptive_system {
	using namespace tensorflow;

	float get_reward(const Tensor& state, const int action_order,
		const float time_interval, const float last_loss, const float current_loss);
}

#endif // !REWARD_H

