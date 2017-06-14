#ifndef ADAPTIVE_SYSTEM_INDEXED_SLICES_H
#define ADAPTIVE_SYSTEM_INDEXED_SLICES_H

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include <functional>
#include <vector>

namespace adaptive_system {
	std::pair<tensorflow::Tensor, tensorflow::Tensor>  merge_multiple_indexed_slices
	(std::vector<std::pair<tensorflow::Tensor, tensorflow::Tensor>> const & vec_ind_slic);
}
#endif // !ADAPTIVE_SYSTEM_INDEXED_SLICES_H
