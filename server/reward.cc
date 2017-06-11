#include "server/reward.h"

namespace adaptive_system {
	using namespace tensorflow;
	//last_loss and current_loss are both moving average losses
	namespace {
		float get_reward_from(const float time_interval, const float last_loss, const float current_loss) {
			return (last_loss - current_loss) / time_interval;
		}
		float get_reward_from_heuristic(const Tensor& state, const GRAD_QUANT_LEVEL action, 
			const float time_interval, const float last_loss, const float current_loss){
			
		float reduction = last_loss - current_loss;
		if (reduction > 0) {
			//good thing happens
			return reduction / time_interval;
		}
		else {
			//bad thing happens
			if (action == GRAD_QUANT_LEVEL::EIGHT) {
				return 0.01;
			}
			else if (action == GRAD_QUANT_LEVEL::SIXTEEN) {
				return 0.015;
			}
			else {
				//reduction < 0 
				return reduction / time_interval * 5;
			}
		}
		}
	}
	float get_reward(const Tensor& state, const GRAD_QUANT_LEVEL action,
		const float time_interval, const float last_loss, const float current_loss) {
		return get_reward_from_heuristic(state, action, time_interval, last_loss, current_loss);
	}
	
}
