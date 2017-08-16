
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
	static float const alpha = 0.1;
	//static int const total_actions = 3;

	namespace {

		void print_state(const Tensor& state) {
			const float* state_ptr = state.flat<float>().data();
			size_t size = state.NumElements();
			for (size_t i = 0; i < size; i++) {
				std::cout << state_ptr[i] << " ";
			}
			std::cout << "\n";
		}
	}

	int sarsa_model::index_of_max(float* array) {
		int index = get_current_index();
		int level = get_current_level();
		if (level == _start_level) {
			if (array[index] > array[index + 1]) {
				return index;
			}
			else {
				return index + 1;
			}
		}
		else if (level == _end_level) {
			if (array[index] > array[index - 1]) {
				return index;
			}
			else {
				return index - 1;
			}
		}
		else {
			if (array[index - 1] > array[index] && array[index - 1] > array[index + 1]) {
				return index - 1;
			}
			else if (array[index + 1] > array[index] && array[index + 1] > array[index - 1]) {
				return index + 1;
			}
			else {
				return index;
			}
		}
	}

	int sarsa_model::get_current_level() {
		return _current_level;
	}

	int sarsa_model::get_current_index() {
		return get_index_from_level(_current_level);
	}

	int sarsa_model::get_total_level_number() {
		return _end_level - _start_level + 1;
	}

	int sarsa_model::get_level_from_index(int index) {
		return _start_level + index;
	}

	int sarsa_model::get_index_from_level(int level) {
		return level - _start_level;
	}

	tensorflow::Tensor sarsa_model::get_feed_tensor_from_index(int index) {
		int total_level_number = get_total_level_number();
		tensorflow::Tensor ret(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({ total_level_number }));
		float* ret_ptr = ret.flat<float>().data();
		std::fill(ret_ptr, ret_ptr + total_level_number, 0.0f);
		ret_ptr[index] = 1.0;
		return ret;
	}

	std::vector<float> sarsa_model::get_greedy_probability(size_t index_of_max) {
		int level_number = get_total_level_number();
		float value = _eps_greedy / 3;
		std::vector<float> ret(level_number, 0);
		if (_current_level == _start_level) {
			if (index_of_max == _current_level) {
				ret[0] = 1 - _eps_greedy + value;
				ret[1] = value * 2;
				return ret;
			}
			else {
				ret[1] = 1 - _eps_greedy + value;
				ret[0] = value * 2;
				return ret;
			}
		}
		else if (_current_level == _end_level) {
			if (index_of_max == _current_level) {
				ret[level_number - 1] = 1 - _eps_greedy + value;
				ret[level_number - 2] = value * 2;
				return ret;
			}
			else {
				ret[level_number - 2] = 1 - _eps_greedy + value;
				ret[level_number - 1] = value * 2;
				return ret;
			}
		}
		else {
			int index = get_current_index();
			ret[index] = ret[index + 1] = ret[index - 1] = value;
			ret[index_of_max] += 1 - -_eps_greedy;
			return ret;
		}
	}

	sarsa_model::sarsa_model(std::string const& path, float r, float eps_greedy, int start, int end, int init)
		: _sarsa_model_path(path), _r(r), _eps_greedy(eps_greedy), _start_level(start),
		_end_level(end), _current_level(init) {
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
		int level) {
		int index = get_index_from_level(level);
		Tensor action_tensor = get_feed_tensor_from_index(index); // one hot vector
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
	//by the way, change the value of _current_level
	int sarsa_model::sample_new_action(Tensor const& state) { // return level
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
		std::vector<float> prob = get_greedy_probability(max_index); //prob.length = _total_level_number
		std::discrete_distribution<int> discrete{ prob.begin(), prob.end() };
		size_t sample = discrete(generator);
		_current_level = get_level_from_index(sample);
		return _current_level;
	}
	//don't change the value of _current_level, all information must be transited in the arguments
	void sarsa_model::adjust_model(float reward, Tensor const& old_state,
		int const old_level,
		Tensor const& new_state,
		int const new_level) {
		print_state(old_state);
		print_state(new_state);
		float old_value = get_q_value(old_state, old_level);
		float new_value = get_q_value(new_state, new_level);
		Tensor learning_rate_tensor(DataType::DT_FLOAT, TensorShape());
		float * learning_rate_ptr = learning_rate_tensor.flat<float>().data();
		*learning_rate_ptr = -alpha * (reward + _r * new_value - old_value);
		std::cout << "old_value: " << old_value << " new_value: " << new_value <<
			" learning_rate: " << *learning_rate_ptr << std::endl;
		int old_index = get_index_from_level(old_level);
		Tensor one_hot_tensor = get_feed_tensor_from_index(old_index);
		Status status = _session->Run({ { state_placeholder_name, old_state },
										{ one_hot_placeholder_name , one_hot_tensor},
										{ learning_rate_placeholder_name, learning_rate_tensor} },
										{}, { training_op_name }, nullptr);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
	}
}
