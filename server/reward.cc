#include "server/reward.h"

namespace adaptive_system {
	using namespace tensorflow;
	//last_loss and current_loss are both moving average losses
	namespace {

		float get_reward_from(const float time_interval, const float last_loss, const float current_loss) {
			return (last_loss - current_loss) / time_interval;
		}

		float get_reward_from_heuristic(const Tensor& state, const int action_order,
			const float time_interval, const float last_loss, const float current_loss) {

			float reduction = last_loss - current_loss;
			//if (reduction > 0) {
			//	//good thing happens
			//	return reduction / time_interval;
			//}
			//else {
			//	//bad thing happens
			//	if (action == GRAD_QUANT_LEVEL::EIGHT) {
			//		return current_loss * 0.1;
			//	}
			//	else if (action == GRAD_QUANT_LEVEL::SIXTEEN) {
			//		return current_loss * 0.15;
			//	}
			//	else {
			//		//reduction < 0 
			//		return reduction / time_interval * 5;
			//	}
			//}
			return reduction / time_interval;
		}

	}

	float get_reward(const Tensor& state, const int action_order,
		const float time_interval, const float last_loss, const float current_loss) {
		return get_reward_from_heuristic(state, action_order, time_interval, last_loss, current_loss);
	}

	float get_reward_v2(float slope) {
		return  -slope;
	}

	float get_reward_v3(float slope) {
		return -slope * 100;
	}

	float get_reward_v4(float slope, int level) {
		return -slope * 100 / level;
	}

	namespace {
		float get_real_size(int level) {
			static const float layer1_size = 0.0192;
			static const float layer2_size = 0.4096;
			static const float layer3_size = 14.450688;
			static const float layer4_size = 8.847360;
			static const float layer5_size = 0.0768;
			float q = 32.0 / level;
			return layer1_size + layer2_size + (layer3_size + layer4_size) / q + layer5_size;
		}
	}

	float get_reward_v5(float slope, int level) {
		float com_oh = get_real_size(level);
		return -slope * 200.0f / com_oh;
	}
}
