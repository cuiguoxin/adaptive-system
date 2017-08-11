#ifndef MULTI_BANDIT_H
#define MULTI_BANDIT_H

#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <functional>
#include <fstream>
#include <map>

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
		multi_bandit(float st_size, float e) : step_size(st_size), eps(e) {
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


template<int start, int end>
class multi_bandit_continous {
public:
	multi_bandit_continous(float st_size, float e, int init_level) : step_size(st_size), eps(e), current_abs_level(init_level) {
		for (int i = start; i <= end; i++) {
			action_value[i] = 0;
		}
	}

	//sample new abs_level
	int sample_new_abs_level() {
		static unsigned seed =
			std::chrono::system_clock::now().time_since_epoch().count();
		static std::default_random_engine generator(seed);
		std::vector<float> prob = get_greedy_probability();
		std::discrete_distribution<int> discrete{ prob.begin(), prob.end() };
		size_t sample = discrete(generator);
		if (current_abs_level == start) {
			current_abs_level += sample;
		}
		else if (current_abs_level == end) {
			current_abs_level -= (1 - sample);
		}
		else {
			current_abs_level -= (1 - sample);
		}
		return current_abs_level;
	}

	void adjust_model(float reward) {
		action_value[current_abs_level] += step_size*(reward - action_value[current_abs_level]);
	}

	void print_value(std::ofstream& stream) {
		for (int i = start; i <= end; i++) {
			stream << std::to_string(action_value[i]) << " ";
		}
		stream << "\n";
	}
private:
	float step_size;
	float eps;
	int current_abs_level;
	std::map<int, float> action_value;

	std::vector<float> get_greedy_probability() {
		std::vector<float> ret;
		if (current_abs_level == start) {
			ret.resize(2, eps / 2);
			if (action_value[current_abs_level] > action_value[current_abs_level + 1]) {
				ret[0] += 1 - eps;
			}
			else {
				ret[1] += 1 - eps;
			}
		}
		else if (current_abs_level == end) {
			ret.resize(2, eps / 2);
			if (action_value[current_abs_level] > action_value[current_abs_level - 1]) {
				ret[1] += 1 - eps;
			}
			else {
				ret[0] += 1 - eps;
			}
		}
		else {
			ret.resize(3, eps / 3);
			float current = action_value[current_abs_level];
			float previous = action_value[current_abs_level - 1];
			float next = action_value[current_abs_level + 1];
			if (current > previous && current > next) {
				ret[1] += 1 - eps;
			}
			else if (previous > current && previous > next) {
				ret[0] += 1 - eps;
			}
			else {
				ret[2] += 1 - eps;
			}
		}
		return ret;
	}

};
#endif // !MULTI_BANDIT_H
