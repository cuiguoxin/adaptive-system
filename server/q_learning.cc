
#include "server/q_learning.h"
#include <cstdlib>
#include <random>

using namespace tensorflow;

namespace adaptive_system {

static std::string const q_values_names = "Reshape:0";
static std::string const state_placeholder_name = "first_layer/state:0";
static std::string const one_hot_placeholder_name = "one_hot:0";
static std::string const action_value_name = "einsum/Reshape_2:0";
static std::string const learning_rate_placeholder_name = "learning_rate:0";
static std::string const training_op_name = "GradientDescent";
static float const alpha = 0.02;
// static int const total_actions = 3;

namespace {

void print_state(const Tensor& state) {
    const float* state_ptr = state.flat<float>().data();
    size_t size = state.NumElements();
    for (size_t i = 0; i < size; i++) {
        std::cout << state_ptr[i] << " ";
    }
    std::cout << "\n";
}
}  // namespace

// keep level continous
int q_learning_model::index_of_max(float* array) {
    const int total = get_total_level_number();
    for (int i = 0; i < total; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
    int const index = get_current_index();

    if (index == (total - 1))
        return index;
    if (array[index] * 1.05 > array[index + 1])
        return index;
    else
        return index + 1;
}
int q_learning_model::index_of_max(tensorflow::Tensor& tensor) {
    float* array = tensor.flat<float>().data();
    const int total = get_total_level_number();
    int const index = get_current_index();

    if (index == (total - 1))
        return index;
    if (array[index] * 1.05 > array[index + 1])
        return index;
    else
        return index + 1;
}

int q_learning_model::get_current_level() {
    return _current_level;
}

int q_learning_model::get_current_index() {
    return get_index_from_level(_current_level);
}

int q_learning_model::get_total_level_number() {
    return _end_level - _start_level + 1;
}

int q_learning_model::get_level_from_index(int index) {
    return _start_level + index;
}

int q_learning_model::get_index_from_level(int level) {
    return level - _start_level;
}

tensorflow::Tensor q_learning_model::get_feed_tensor_from_index(int index) {
    int total_level_number = get_total_level_number();
    tensorflow::Tensor ret(tensorflow::DataType::DT_FLOAT,
                           tensorflow::TensorShape({total_level_number}));
    float* ret_ptr = ret.flat<float>().data();
    std::fill(ret_ptr, ret_ptr + total_level_number, 0.0f);
    ret_ptr[index] = 1.0;
    return ret;
}

q_learning_model::q_learning_model(std::string const& path,
                                   int const input_size,
                                   float r,
                                   float eps_greedy,
                                   int start,
                                   int end,
                                   int init)
    : _q_learning_model_path(path),
      _r(r),
      _eps_greedy(eps_greedy),
      _start_level(start),
      _end_level(end),
      _current_level(init),
      _discrete({1.0 - _eps_greedy, _eps_greedy}) {
    _session = NewSession(SessionOptions());
    GraphDef graph_def;
    // may first generate the .pb file
    int output_size = end - start + 1;
    std::string command =
        "python /home/cgx/git_project/"
        "adaptive-system/reinforcement_learning_model/"
        "q_learning_continuous.py " +
        std::to_string(input_size) + " " + std::to_string(output_size);
    int error_code = system(command.c_str());
    if (error_code != 0) {
        PRINT_ERROR_MESSAGE(
            "python q_learning_continous.py failed and error code is " +
            std::to_string(error_code));
        // std::terminate();
    }
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
    status = _session->Run({}, {}, {"init"}, nullptr);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
}

float q_learning_model::get_q_value(tensorflow::Tensor const& state,
                                    int level) {
    int index = get_index_from_level(level);
    Tensor action_tensor = get_feed_tensor_from_index(index);  // one hot vector
    std::vector<Tensor> result;
    Status status = _session->Run({{state_placeholder_name, state},
                                   {one_hot_placeholder_name, action_tensor}},
                                  {action_value_name}, {}, &result);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
    Tensor& result_tensor = result[0];
    float* ret = result_tensor.flat<float>().data();
    float ret_v = *ret;
    return ret_v;
}

int q_learning_model::sample() {
    static unsigned seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    static std::default_random_engine generator(seed);
    return _discrete(generator);
}

// by the way, change the value of _current_level
int q_learning_model::sample_new_action() {  // return level
    int const num_rand = sample();
    if (num_rand == 0) {
        return _current_level;
    } else {
        return _current_level + 1;
    }
}
void q_learning_model::transform_state(Tensor const& state) {
    std::vector<Tensor> result;
    Status status = _session->Run({{state_placeholder_name, state}},
                                  {q_values_names}, {}, &result);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
    Tensor& result_tensor = result[0];
    float* result_tensor_ptr = result_tensor.flat<float>().data();
    int const max_index = index_of_max(result_tensor_ptr);
    std::cout << "max index is: " << max_index << std::endl;
    int const index = get_current_index();
    if (max_index == index + 1) {
        _current_level++;
    }
}

float q_learning_model::max_q_learning_value(Tensor const& state) {
    std::vector<Tensor> result;
    Status status = _session->Run({{state_placeholder_name, state}},
                                  {q_values_names}, {}, &result);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
    Tensor& result_tensor = result[0];
    float* result_tensor_ptr = result_tensor.flat<float>().data();
    int const max_index = index_of_max(result_tensor_ptr);
    return result_tensor_ptr[max_index];
}

// don't change the value of _current_level, all information must be transited
// in the arguments
void q_learning_model::adjust_model(float reward,
                                    Tensor const& old_state,
                                    int const old_level,
                                    Tensor const& new_state) {
    // print_state(old_state);
    // print_state(new_state);
    float const old_value = get_q_value(old_state, old_level);
    // int const new_index = index_of_max(new_state);
    // int const new_level = get_level_from_index(new_index);
    float const max_value = max_q_learning_value(new_state);
    // learning rate construction
    Tensor learning_rate_tensor(DataType::DT_FLOAT, TensorShape());
    float* learning_rate_ptr = learning_rate_tensor.flat<float>().data();
    *learning_rate_ptr = -alpha * (reward + _r * max_value - old_value);

    std::cout << "old_value: " << old_value << " max_value: " << max_value
              << " learning_rate: " << *learning_rate_ptr << std::endl;
    std::cout << "Q(s, " << old_level << ") = Q(s, " << old_level << ") + "
              << alpha << " * (" << reward << " + " << _r << " * " << max_value
              << " - Q(s, " << old_level << ")" << std::endl;
    int old_index = get_index_from_level(old_level);
    Tensor one_hot_tensor = get_feed_tensor_from_index(old_index);
    Status status =
        _session->Run({{state_placeholder_name, old_state},
                       {one_hot_placeholder_name, one_hot_tensor},
                       {learning_rate_placeholder_name, learning_rate_tensor}},
                      {}, {training_op_name}, nullptr);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
}
}  // namespace adaptive_system
