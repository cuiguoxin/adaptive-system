
#include "server/sarsa.h"

using namespace tensorflow;

namespace adaptive_system {
	sarsa_model::sarsa_model(std::string const & path, float r, float eps_greedy) :_sarsa_model_path(path),
		_r(r), _eps_greedy(eps_greedy) {
		_session = NewSession(SessionOptions());
		GraphDef graph_def;
		Status status = ReadBinaryProto(Env::Default(), path, &graph_def);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}

		// Add the graph to the session
		status = _session->Create(graph_def);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
		//init all the variable
		status = _session->Run({}, {}, { "init" }, nullptr);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}

	}

	float sarsa_model::get_q_value(tensorflow::Tensor const & state, GRAD_QUANT_LEVEL action) {
		static std::string const state_placeholder_name = "";
		static std::string const one_hot_placeholder_name = "";
		static std::string const action_value_name = "";
		Tensor action_tensor = get_feed_tensor_from_action(action);
		std::vector<Tensor> result;
		Status status = _session->Run({ {state_placeholder_name, state}, {one_hot_placeholder_name, action_tensor} },
			{action_value},
			{},
			&result);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
		Tensor& result_tensor = result[0];
		float* ret = result_tensor.flat<float>().data();
		float ret_v = *ret;
		return ret_v;
	}

	GRAD_QUANT_LEVEL sarsa_model::sample_new_action() {

	}
	void sarsa::adjust_model(float reward,
		tensorflow::Tensor const& old_state,
		GRAD_QUANT_LEVEL old_action,
		tensorflow::Tensor const& new_state,
		GRAD_QUANT_LEVEL new_action) {

	}
}
