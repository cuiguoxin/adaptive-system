#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

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

namespace input {
	using namespace tensorflow;
	Session* load_graph_and_create_session(const std::string& graph_path) {
		GraphDef graph_def;
		Status status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
		if (!status.ok()) {
			std::cout << status.ToString() << "\n";
			std::terminate();
		}
		Session* session;
		status = NewSession(SessionOptions(), &session);
		if (!status.ok()) {
			std::cout << status.ToString() << "\n";
			std::terminate();
		}
		status = session->Create(graph_def);
		if (!status.ok()) {
			std::cout << status.ToString() << "\n";
			std::terminate();
		}
		return session;
	}
}