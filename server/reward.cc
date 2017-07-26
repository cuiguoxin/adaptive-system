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
			return reduction / time_interval;
		}

	}

	float get_reward(const Tensor& state, const int action_order,
		const float time_interval, const float last_loss, const float current_loss) {
		return get_reward_from_heuristic(state, action_order, time_interval, last_loss, current_loss);
	}

}
