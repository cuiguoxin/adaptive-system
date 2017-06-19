#ifndef WORD2VEC_INPUT_H
#define WORD2VEC_INPUT_H

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
namespace adaptive_system {
	namespace word2vec {
		void init(std::string const & raw_data_path);
		std::pair<tensorflow::Tensor, tensorflow::Tensor> get_next_batch(size_t const batch_size);
	}

}

#endif // !WORD2VEC_INPUT_H
