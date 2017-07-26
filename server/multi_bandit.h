#ifndef MULTI_BANDIT_H
#define MULTI_BANDIT_H

#include <chrono>
#include <random>
#include <vector>

namespace adaptive_system {
	template <int action_size, float step_size, float eps>
	class multi_bandit {
	private:
		float action_value[action_size];
		std::vector<float> get_greedy_probability() {
			int index = 0; 
			float max = action_value[index];
			for (int i = 0; i < action_size; i++) {
				if (max < action_value[i]) {
					index = i;
					max = action_value[i];
				}
			}
			std::vector<float> ret_vec;
			ret_vec.resize(action_size, eps / action_size);
			ret_vec[index] += 1 - eps;
			return ret_vec;
		}
	public:
		int sample_action() {
			static unsigned seed =
				std::chrono::system_clock::now().time_since_epoch().count();
			static std::default_random_engine generator(seed);
			std::vector<float> prob = get_greedy_probability();
			std::discrete_distribution<int> discrete{ prob.begin(), prob.end() };
			size_t sample = discrete(generator);
			return sample;
		}
		void adjust_model(float reward, int action_index) {
			action_value[action_index] += step_size * (reward - action_value[action_index]);
		}
	};
}
#endif // !MULTI_BANDIT_H
