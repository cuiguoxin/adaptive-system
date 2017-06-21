
#include "server/sarsa.h"
#include <random>

using namespace tensorflow;

namespace adaptive_system {

	static std::string const q_values_names = "Reshape:0";
	static std::string const state_placeholder_name = "first_layer/state:0";
	static std::string const one_hot_placeholder_name = "one_hot:0";
	static std::string const action_value_name = "Reshape_3:0";
	static std::string const learning_rate_placeholder_name = "learning_rate:0";
	static std::string const training_op_name = "GradientDescent";
	static size_t const total_actions = 5;
	static size_t const total_features = 8;
	static float const alpha = 0.1;

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
		void print_state(const Tensor& state) {
			const float* state_ptr = state.flat<float>().data();
			for (size_t i = 0; i < total_features; i++) {
				std::cout << state_ptr[i] << " ";
			}
			std::cout << "\n";
		}
	}

	std::vector<float> sarsa_model::get_greedy_probability(size_t index_of_max) {
		float value = _eps_greedy / total_actions;
		std::vector<float> ret(total_actions, value);
		ret[index_of_max] += 1 - _eps_greedy;
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
		int action_order) {
		Tensor action_tensor = get_feed_tensor_from_action(action_order);
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

	int sarsa_model::sample_new_action(Tensor const& state) {
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
		return sample;
	}

	void sarsa_model::adjust_model(float reward, Tensor const& old_state,
		int const old_action_order,
		Tensor const& new_state,
		int const new_action_order) {
		print_state(old_state);
		print_state(new_state);
		float old_value = get_q_value(old_state, old_action_order);
		float new_value = get_q_value(new_state, new_action_order);
		float update = reward + _r * new_value - old_value;
		Tensor learning_rate_tensor(DataType::DT_FLOAT, TensorShape());
		float * learning_rate_ptr = learning_rate_tensor.flat<float>().data();
		*learning_rate_ptr = -alpha * (reward + _r * new_value - old_value);
		std::cout << "old_value: " << old_value << " new_value: " << new_value <<
			" learning_rate: " << *learning_rate_ptr << std::endl;
		Tensor one_hot_tensor = get_feed_tensor_from_action(old_action_order);
		Status status = _session->Run({ { state_placeholder_name, old_state }, 
										{ one_hot_placeholder_name , one_hot_tensor},
										{ learning_rate_placeholder_name, learning_rate_tensor} },
										{}, {training_op_name}, nullptr);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
	}
}
