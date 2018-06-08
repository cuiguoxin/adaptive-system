#include "server/reward.h"

namespace adaptive_system {
using namespace tensorflow;
// last_loss and current_loss are both moving average losses
namespace {

float get_reward_from(const float time_interval,
                      const float last_loss,
                      const float current_loss) {
    return (last_loss - current_loss) / time_interval;
}

float get_reward_from_heuristic(const Tensor& state,
                                const int action_order,
                                const float time_interval,
                                const float last_loss,
                                const float current_loss) {
    float reduction = last_loss - current_loss;
    return reduction / time_interval;
}

}  // namespace

float get_reward(const Tensor& state,
                 const int action_order,
                 const float time_interval,
                 const float last_loss,
                 const float current_loss) {
    return get_reward_from_heuristic(state, action_order, time_interval,
                                     last_loss, current_loss);
}

float get_reward_v2(float slope) {
    return -slope;
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
    return layer1_size + layer2_size + (layer3_size + layer4_size) / q +
           layer5_size;
}
std::vector<float> time_table;
class TimeTable {
   public:
    TimeTable(float const computing_time,
              float const one_bit_communication_time) {
        time_table.resize(33);
        // set computing time
        time_table[0] = computing_time;
        // set all level
        int const size = time_table.size();
        for (int i = 1; i < size; i++) {
            time_table[i] = time_table[i - 1] + one_bit_communication_time;
        }
    }
};
}  // namespace

float get_reward_v5(float const slope,
                    int const level,
                    float const computing_time,
                    float const one_bit_communication_time) {
    static TimeTable tt(computing_time, one_bit_communication_time);
    // float com_oh = get_real_size(level);
    const float time = time_table[level];
    return -slope * 300.0f / time;
}
}  // namespace adaptive_system
