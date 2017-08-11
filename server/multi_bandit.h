#ifndef MULTI_BANDIT_H
#define MULTI_BANDIT_H

#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <functional>
#include <fstream>

namespace adaptive_system {
	template <int action_size>
	class multi_bandit {
	private:
		float step_size;
		float eps;
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
		multi_bandit(float st_size, float e): step_size(st_size), eps(e) {
			std::fill(action_value, action_value + action_size, 0.0f);
		}

		int sample_new_action() {
			static unsigned seed =
				std::chrono::system_clock::now().time_since_epoch().count();
			static std::default_random_engine generator(seed);
			std::vector<float> prob = get_greedy_probability();
			std::discrete_distribution<int> discrete{ prob.begin(), prob.end() };
			size_t sample = discrete(generator);
			return sample;
		}

		void print_value(std::ofstream& stream) {
			for (int i = 0; i < action_size; i++) {
				stream << std::to_string(action_value[i]) << " ";
			}
			stream << "\n";
		}

		void adjust_model(float reward, int action_index) {
			if (action_index == 1) {
				action_value[action_index] += step_size * (reward - action_value[action_index]);
			}
			else {
				action_value[action_index] += step_size * (reward - action_value[action_index]);
				action_value[1] += step_size * (reward - action_value[1]);
			}
			
		}
	};
}
#endif // !MULTI_BANDIT_H
