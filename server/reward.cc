#include "server/reward.h"

namespace adaptive_system {
	using namespace tensorflow;
	//last_loss and current_loss are both moving average losses
	float get_reward(const Tensor& state, const GRAD_QUANT_LEVEL action,
		const float time_interval, const float last_loss, const float current_loss) {
		float reduction = last_loss - current_loss;
		if (reduction > 0) {
			//good thing happens
			return reduction / time_interval;
		}
		else {
			//bad thing happens
			if (action == GRAD_QUANT_LEVEL::EIGHT) {
				return 0.05;
			}
			else if (action = GRAD_QUANT_LEVEL::SIXTEEN) {
				return 0.07;
			}
			else {
				//reduction < 0 
				return reduction / time_interval;
			}
		}
	}
}