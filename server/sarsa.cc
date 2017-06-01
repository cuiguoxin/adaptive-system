
#include "server/sarsa.h"
#include <random>

using namespace tensorflow;

namespace adaptive_system {

	static std::string const q_values_names = "";
	static std::string const state_placeholder_name = "";
	static std::string const one_hot_placeholder_name = "";
	static std::string const action_value_name = "";
	static size_t const total_actions = 5;
	static size_t const total_features = 7;
	static float const alpha = 0.01;

	namespace {
		size_t index_of_max(float* array) {
			size_t index = 0;
			float max = array[0];
			for (size_t i = 1; i < total_actions; i++) {
				if (max < array[i]) {
					max = array[i];
					index = i;
				}
			}
			return index;
		}
	}

	Tensor sarsa_model::get_feed_tensor_from_action(GRAD_QUANT_LEVEL action) {
		Tensor ret(DataType::DT_FLOAT, TensorShape({ total_features });
		float* ret_ptr = ret.flat<float>().data();
		std::fill(ret_ptr, ret_ptr + total_features, 0.0f);
		switch (action) {
		case GRAD_QUANT_LEVEL::ONE:
			ret_ptr[0] = 1.0f;
			break;
		case GRAD_QUANT_LEVEL::TWO:
			ret_ptr[1] = 1.0f;
			break;
		case GRAD_QUANT_LEVEL::FOUR:
			ret_ptr[2] = 1.0f;
			break;
		case GRAD_QUANT_LEVEL::EIGHT:
			ret_ptr[3] = 1.0f;
			break;
		case GRAD_QUANT_LEVEL::SIXTEEN:
			ret_ptr[4] = 1.0f;
			break;
		}
		return ret;
	}

	std::vector<float> sarsa_model::get_greedy_probability(size_t index_of_max) {
		float value = _eps_greedy / total_actions;
		std::vector<float> ret(total_actions, value);
		ret[index_of_max] += value;
		return ret;
	}
	sarsa_model::sarsa_model(std::string const& path, float r, float eps_greedy)
		: _sarsa_model_path(path), _r(r), _eps_greedy(eps_greedy) {
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
		// init all the variable
		status = _session->Run({}, {}, { "init" }, nullptr);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
	}

	float sarsa_model::get_q_value(tensorflow::Tensor const& state,
		GRAD_QUANT_LEVEL action) {
		Tensor action_tensor = get_feed_tensor_from_action(action);
		std::vector<Tensor> result;
		Status status = _session->Run({ {state_placeholder_name, state},
									   {one_hot_placeholder_name, action_tensor} },
									   { action_value_name }, {}, &result);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
		Tensor& result_tensor = result[0];
		float* ret = result_tensor.flat<float>().data();
		float ret_v = *ret;
		return ret_v;
	}

	GRAD_QUANT_LEVEL sarsa_model::sample_new_action(Tensor const& state) {
		static unsigned seed =
			std::chrono::system_clock::now().time_since_epoch().count();
		static std::default_random_engine generator(seed);
		std::vector<Tensor> result;
		Status status = _session->Run({ {state_placeholder_name, state} },
		{ q_values_names }, {}, &result);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
		Tensor& result_tensor = result[0];
		float* result_tensor_ptr = result_tensor.flat<float>().data();
		size_t max_index = index_of_max(result_tensor_ptr);
		std::vector<float> prob = get_greedy_probability(max_index);
		std::discrete_distribution<int> discrete{ prob.begin(), prob.end() };
		size_t sample = discrete(generator);
		switch (sample) {
		case 0:
			return GRAD_QUANT_LEVEL::ONE;
		case 1:
			return GRAD_QUANT_LEVEL::TWO;
		case 2:
			return GRAD_QUANT_LEVEL::FOUR;
		case 3:
			return GRAD_QUANT_LEVEL::EIGHT;
		case 4:
			return GRAD_QUANT_LEVEL::SIXTEEN;
		}
		std::terminate();
		return nullptr;
	}

	void sarsa::adjust_model(float reward, tensorflow::Tensor const& old_state,
		GRAD_QUANT_LEVEL old_action,
		Tensor const& new_state,
		GRAD_QUANT_LEVEL new_action) {
		float old_value = get_q_value(old_state, old_action);
		float new_value = get_q_value(new_state, new_action);
		float update = reward + _r * new_value - old_value;
		Tensor learning_rate_tensor(DataType::DT_FLOAT, TensorShape());
		float * learning_rate_ptr = learning_rate_ptr.flat<float>().data();
		
		Status status = _session->Run({}, {}, {}, nullptr);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
	}
}
