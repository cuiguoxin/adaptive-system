#include "server/actor_critic.h"
#include <random>

using namespace tensorflow;

namespace adaptive_system {
	namespace {
		//learning_rate
		const std::string policy_learning_rate_name = "";
		const std::string value_function_learning_rate_name = "";
		//init
		const std::string init_name = "";
		//state placeholder
		const std::string one_hot_state_name = "";
		const std::string policy_state_name = "";
		const std::string value_state_name = "";
		//value name
		const std::string value_name = "";
		//policy name, whole vector name, length is 5, not 1
		const std::string policy_value_name = ""; 
		//training op name
		const std::string policy_training_name = "";
		const std::string value_training_name = "";
		const size_t action_number = 5;
		const size_t state_number = 8;
	}
	actor_critic::actor_critic(std::string const & model_path, float const r, float const beta, size_t t)
		:_sarsa_model_path(model_path), _r(r), _beta(beta), _T(t) {
		_session = NewSession(SessionOptions());
		GraphDef graph_def;
		Status status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
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
		status = _session->Run({}, {}, { init_name }, nullptr);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
	}

	GRAD_QUANT_LEVEL actor_critic::sample_action_from_policy(tensorflow::Tensor const & state) {
		static unsigned seed =
			std::chrono::system_clock::now().time_since_epoch().count();
		static std::default_random_engine generator(seed);
		std::vector<Tensor> result;
		Status status = _session->Run({ { policy_state_name, state } },
		{ policy_value_name }, {}, &result);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
		Tensor& result_tensor = result[0];
		float* result_tensor_ptr = result_tensor.flat<float>().data();
		std::discrete_distribution<int> discrete{ result_tensor_ptr, result_tensor_ptr + action_number };
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
		return GRAD_QUANT_LEVEL::NONE;
	}

	float actor_critic::get_value(tensorflow::Tensor const & state) {
		std::vector<Tensor> result;
		Status status = _session->Run({ { value_state_name, state } },
		{ value_name }, {}, &result);
		if (!status.ok()) {
			PRINT_ERROR_MESSAGE(status.error_message());
			std::terminate();
		}
		Tensor& result_tensor = result[0];
		float* result_tensor_ptr = result_tensor.flat<float>().data();
		return *result_tensor_ptr;
	}

	float actor_critic::get_update_value(float reward, tensorflow::Tensor const & new_state,
		tensorflow::Tensor const & last_state) {
		float new_value = get_value(new_state);
		float last_value = get_value(last_state);
		float ret = reward + _r * new_value - last_value;
	}
	void actor_critic::update_value_function_parameter(tensorflow::Tensor const & state,
		const float update) {

	}
	void actor_critic::update_policy_parameter(tensorflow::Tensor const & state, const float update) {

	}


}