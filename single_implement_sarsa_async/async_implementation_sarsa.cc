#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#include "quantization/util/algorithms.h"
#include "quantization/util/any_level.h"
#include "quantization/util/helper.h"
#include "server/reward.h"
#include "server/sarsa.h"
#include "single_implement_sarsa_async/accuracy.h"

namespace input {
using namespace tensorflow;

unsigned int index_current = 0;
int batch_size;
std::vector<Tensor> raw_tensors, standard_images, standard_labels;
const int record_size = 3073;
const int label_size = 1;
const int image_size = 3072;

namespace {
Session* load_graph_and_create_session(const std::string& graph_path) {
    GraphDef graph_def;
    Status status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        std::terminate();
    }
    Session* session;
    status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        std::terminate();
    }
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        std::terminate();
    }
    return session;
}
void read_raw_tensors_from_file(const std::string& binary_file_prefix) {
    for (int i = 1; i <= 5; i++) {
        std::ifstream input_stream(
            binary_file_prefix + std::to_string(i) + ".bin", std::ios::binary);
        TensorShape raw_tensor_shape({record_size});
        if (input_stream.is_open()) {
            for (int j = 0; j < 10000; j++) {
                Tensor raw_tensor(DataType::DT_UINT8, raw_tensor_shape);
                uint8* raw_tensor_ptr = raw_tensor.flat<uint8>().data();
                input_stream.read(reinterpret_cast<char*>(raw_tensor_ptr),
                                  record_size);
                raw_tensors.push_back(raw_tensor);
            }
        }
        input_stream.close();
    }
    PRINT_INFO;
    // shuffle the vector raw_tensors
    /*unsigned seed =
    std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(raw_tensors.begin(), raw_tensors.end(),
            std::default_random_engine(seed));*/
}
}  // namespace

void turn_raw_tensors_to_standard_version(
    const std::string& binary_file_prefix =
        "/home/cgx/git_project/"
        "adaptive-system/resources/"
        "cifar-10-batches-bin/data_batch_",
    const std::string& preprocess_graph_path =
        "/home/cgx/git_project/adaptive-system/input/cifar10/preprocess.pb") {
    PRINT_INFO;
    Session* session = load_graph_and_create_session(preprocess_graph_path);
    PRINT_INFO;
    read_raw_tensors_from_file(binary_file_prefix);
    std::cout << raw_tensors.size() << std::endl;
    for (int i = 0; i < 50000; i++) {
        Tensor raw_tensor = raw_tensors[i];
        std::vector<Tensor> image_and_label;
        Status status = session->Run({{"raw_tensor", raw_tensor}},
                                     {"div", "label"}, {}, &image_and_label);
        if (!status.ok()) {
            std::cout << "failed in line " << __LINE__ << " in file "
                      << __FILE__ << " " << status.error_message() << std::endl;
            std::terminate();
        }
        standard_images.push_back(image_and_label[0]);
        standard_labels.push_back(image_and_label[1]);
    }
    raw_tensors.clear();
    PRINT_INFO;
}

std::pair<Tensor, Tensor> get_next_batch() {
    static std::mutex mu;
    int standard_images_size = 3 * 28 * 28;
    TensorShape images_batch_shape({batch_size, 28, 28, 3}),
        labels_batch_shape({batch_size});
    Tensor images_batch(DataType::DT_FLOAT, images_batch_shape),
        labels_batch(DataType::DT_INT32, labels_batch_shape);
    float* images_batch_ptr = images_batch.flat<float>().data();
    int* label_batch_ptr = labels_batch.flat<int>().data();
    std::unique_lock<std::mutex> lk(mu);
    for (int i = 0; i < batch_size; i++) {
        int real_index = index_current % 50000;
        Tensor& image_current = standard_images[real_index];
        float* image_current_ptr = image_current.flat<float>().data();
        std::copy(image_current_ptr, image_current_ptr + standard_images_size,
                  images_batch_ptr + i * standard_images_size);
        Tensor& label_current = standard_labels[real_index];
        int* label_current_ptr = label_current.flat<int>().data();
        label_batch_ptr[i] = *label_current_ptr;
        index_current++;
    }
    lk.unlock();
    return std::pair<Tensor, Tensor>(images_batch, labels_batch);
}
}  // namespace input

namespace client {

using namespace tensorflow;
using namespace adaptive_system;

Tuple tuple;
std::string batch_placeholder_name;
std::string label_placeholder_name;
const int threshold_to_quantize = 105000;

namespace {
// return {loss_total, loss_entropy}
std::pair<float, float> compute_gradient_and_loss(
    tensorflow::Session* session_local,
    std::vector<std::pair<std::string, tensorflow::Tensor>> feeds,
    std::map<std::string, tensorflow::Tensor>& gradients) {
    std::vector<std::string> fetch;
    std::string loss_name = tuple.loss_name();
    auto entropy_loss = tuple.cross_entropy_loss_name();
    fetch.push_back(loss_name);
    fetch.push_back(entropy_loss);
    std::vector<tensorflow::Tensor> outputs;
    std::vector<std::string> variable_names_in_order;
    google::protobuf::Map<std::string, Names> const& map_names =
        tuple.map_names();
    std::for_each(
        map_names.begin(), map_names.end(),
        [&fetch, &variable_names_in_order](
            google::protobuf::MapPair<std::string, Names> const& pair) {
            Names const& names = pair.second;
            std::string const& variable_name = pair.first;
            fetch.push_back(names.gradient_name());
            variable_names_in_order.push_back(variable_name);
        });
    tensorflow::Status tf_status =
        session_local->Run(feeds, fetch, {}, &outputs);
    if (!tf_status.ok()) {
        PRINT_ERROR_MESSAGE(tf_status.error_message());
        std::terminate();
    }
    tensorflow::Tensor& loss_tensor = outputs[0];
    float* loss_ptr = loss_tensor.flat<float>().data();
    float loss_ret = loss_ptr[0];

    auto& entropy_tensor = outputs[1];
    float* entropy_ptr = entropy_tensor.flat<float>().data();
    float loss_entropy_ret = entropy_ptr[0];

    outputs.erase(outputs.begin());
    outputs.erase(outputs.begin());

    size_t size = outputs.size();
    for (size_t i = 0; i < size; i++) {
        gradients.insert(std::pair<std::string, tensorflow::Tensor>(
            variable_names_in_order[i], outputs[i]));
    }
    return {loss_ret, loss_entropy_ret};
}
}  // namespace

tensorflow::Session* load_primary_model_on_master_and_init(
    std::string const tuple_local_path) {
    tensorflow::Session* session_local =
        tensorflow::NewSession(tensorflow::SessionOptions());
    std::fstream input(tuple_local_path, std::ios::in | std::ios::binary);

    if (!input) {
        std::cout << tuple_local_path
                  << ": File not found.  Creating a new file." << std::endl;
    } else if (!tuple.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse tuple." << std::endl;
        std::terminate();
    }
    input.close();
    GraphDef graph_def = tuple.graph();
    Status tf_status = session_local->Create(graph_def);
    if (!tf_status.ok()) {
        std::cout << "create graph has failed in line " << __LINE__
                  << " in file " << __FILE__ << std::endl;
        std::terminate();
    }

    // init parameters
    std::string init_name = tuple.init_name();
    std::cout << init_name << std::endl;
    tf_status = session_local->Run({}, {}, {init_name}, nullptr);
    if (!tf_status.ok()) {
        std::cout << "running init has  failed in line " << __LINE__
                  << " in file " << __FILE__ << std::endl;
        std::terminate();
    }
    // init some names
    batch_placeholder_name = tuple.batch_placeholder_name();
    label_placeholder_name = tuple.label_placeholder_name();
    return session_local;
}

tensorflow::Session* load_primary_model_on_client_and_init(
    tensorflow::Session* session_master) {
    tensorflow::Session* session_local =
        tensorflow::NewSession(tensorflow::SessionOptions());
    GraphDef graph_def = tuple.graph();
    Status tf_status = session_local->Create(graph_def);
    if (!tf_status.ok()) {
        std::cout << "create graph has failed in line " << __LINE__
                  << " in file " << __FILE__ << std::endl;
        std::terminate();
    }
    // init parameters
    std::string init_name = tuple.init_name();
    std::cout << init_name << std::endl;
    tf_status = session_local->Run({}, {}, {init_name}, nullptr);
    if (!tf_status.ok()) {
        std::cout << "running init has  failed in line " << __LINE__
                  << " in file " << __FILE__ << std::endl;
        std::terminate();
    }
    adaptive_system::copy_variable_between_session(session_master,
                                                   session_local, tuple);
    return session_local;
}

void compute_gradient_loss_and_quantize(
    tensorflow::Session* session_local,
    const int level,
    std::map<std::string, tensorflow::Tensor>& map_gradients,
    std::pair<float, float>& loss) {
    // PRINT_INFO;
    std::pair<tensorflow::Tensor, tensorflow::Tensor> feeds =
        input::get_next_batch();
    // PRINT_INFO;
    loss = compute_gradient_and_loss(session_local,
                                     {{batch_placeholder_name, feeds.first},
                                      {label_placeholder_name, feeds.second}},
                                     map_gradients);
    // PRINT_INFO;
    NamedGradientsAccordingColumn named_gradients_send;
    quantize_gradients_according_column(map_gradients, &named_gradients_send,
                                        level, threshold_to_quantize);
    map_gradients.clear();
    dequantize_gradients_according_column(named_gradients_send, map_gradients);
}
}  // namespace client

namespace logging {

std::ofstream file_loss_stream;

void init_log(int const level,
              int const total_worker_num,
              int const sarsa_interval,
              int interval_to_change_variable) {
    // init log
    auto now = std::chrono::system_clock::now();
    auto init_time_t = std::chrono::system_clock::to_time_t(now);
    std::string label = std::to_string(init_time_t);
    std::string store_loss_file_path =
        "single_sarsa_async" + label + "_level:" + std::to_string(level) +
        "_number_of_workers:" + std::to_string(total_worker_num) +
        "_sarsaInterval:" + std::to_string(sarsa_interval) +
        "_intervalExchangeVariable:" +
        std::to_string(interval_to_change_variable);
    file_loss_stream.open("loss_result/" + store_loss_file_path);
    // init predict
    // init(store_loss_file_path);
}

inline void log_to_file(float const time,
                        float const loss,
                        float const cross_entropy_loss,
                        int const current_iter,
                        int const current_level) {
    file_loss_stream << std::to_string(time)
                     << ":: iter num ::" << std::to_string(current_iter)
                     << ":: loss is ::" << loss << "::" << current_level
                     << ":: cross_entropy_loss is :: " << cross_entropy_loss
                     << "\n";
    file_loss_stream.flush();
}
}  // namespace logging

namespace sarsa {
tensorflow::Tensor last_state;
std::vector<float> loss_history;
float _last_loss = 10.0f;
float computing_time = 0.0f;
float one_bit_communication_time = 0.0f;

volatile int& get_current_level() {
    static volatile int level = 2;
    return level;
}

void adjust_rl_model(adaptive_system::sarsa_model& sm, int& level) {
    using namespace adaptive_system;
    auto start = std::chrono::system_clock::now();

    std::vector<float> moving_average_losses;
    const float r = 0.99;
    _last_loss = moving_average_from_last_loss(_last_loss, loss_history,
                                               moving_average_losses, r);
    tensorflow::Tensor new_state =
        get_float_tensor_from_vector(moving_average_losses);
    int new_level = sm.sample_new_action(new_state);
    int old_level = level;

    float slope = get_slope_according_loss(moving_average_losses);
    std::cout << "slope is " << slope << " new level is " << new_level
              << std::endl;
    float reward =
        get_reward_v5(slope, old_level, computing_time,
                      one_bit_communication_time);  // -slope * 100 / level
    sm.adjust_model(reward, last_state, old_level, new_state, new_level);
    level = new_level;
    last_state = new_state;
    loss_history.clear();
    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "RL model spend "
              << double(duration.count()) *
                     std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den
              << "s" << std::endl;
}
}  // namespace sarsa

namespace job {

float computing_time = 0.0f;
float one_bit_communication_time = 0.0f;
std::mutex job_update_mutex;
int current_iter_num = 0;

void do_work_for_one_worker(
    const int worker_id,
    int const total_iter_num_to_run,
    int const level,
    tensorflow::Session* session_local,  // has been initialized
    tensorflow::Session* session_master,
    float const learning_rate_value) {
    for (int i = 0; i < total_iter_num_to_run; i++) {
        std::map<std::string, tensorflow::Tensor> gradients;
        std::pair<float, float> loss_results;
        client::compute_gradient_loss_and_quantize(session_local, level,
                                                   gradients, loss_results);
        // mutex lock
        std::unique_lock<std::mutex> lk(job_update_mutex);
        std::cout << "iter_num: " << current_iter_num
                  << " loss: " << loss_results.first
                  << " entropy_loss:" << loss_results.second << std::endl;
        logging::log_to_file(0.0f, loss_results.first, loss_results.second,
                             current_iter_num++, level);
        adaptive_system::apply_quantized_gradient_to_model(
            gradients, session_master, client::tuple, learning_rate_value);
        adaptive_system::copy_variable_between_session(
            session_master, session_local, client::tuple);
        lk.unlock();
    }
}
// only copy gradient between worker and server
std::map<int, int>& id_to_last_iter() {
    static std::map<int, int> id2last_iter;
    return id2last_iter;
}

std::map<int, int>& worker_id_to_its_iter() {
    static std::map<int, int> worker_id_2_its_iter;
    return worker_id_2_its_iter;
}

void do_work_for_one_worker_v2(
    const int worker_id,
    int const total_iter_num_to_run,
    int const level,
    int const total_worker,
    tensorflow::Session* session_local,  // has been initialized
    tensorflow::Session* session_master,
    float const learning_rate_value,
    int const start_iter) {
    for (int i = 0; i < total_iter_num_to_run; i++) {
        std::map<std::string, tensorflow::Tensor> gradients;
        std::pair<float, float> loss_results;
        client::compute_gradient_loss_and_quantize(session_local, level,
                                                   gradients, loss_results);
        // mutex lock
        std::unique_lock<std::mutex> lk(job_update_mutex);
        std::cout << "iter_num: " << current_iter_num
                  << " loss: " << loss_results.first
                  << " entropy_loss:" << loss_results.second << std::endl;
        logging::log_to_file(0.0f, loss_results.first, loss_results.second,
                             current_iter_num, level);
        adaptive_system::apply_quantized_gradient_to_model(
            gradients, session_master, client::tuple, learning_rate_value);
        // adaptive_system::copy_variable_between_session(
        //     session_master, session_local, client::tuple);
        auto& id2last_iter = id_to_last_iter();
        auto& iter2gradient = adaptive_system::iter_to_gradient();
        iter2gradient[current_iter_num] = {gradients, 0};
        int last_iter;
        auto iter = id2last_iter.find(worker_id);
        if (iter == id2last_iter.end()) {
            last_iter = start_iter - 1;
        } else {
            last_iter = iter->second;
        }
        adaptive_system::copy_gradient_between_session(
            last_iter, current_iter_num, total_worker, level,
            learning_rate_value, session_master, session_local, client::tuple,
            client::threshold_to_quantize);
        id2last_iter[worker_id] = current_iter_num;
        current_iter_num++;
        lk.unlock();
    }
}

void do_work_for_one_worker_v3(
    const int worker_id,
    int const total_iter_num_to_run,
    int const total_worker,
    tensorflow::Session* session_local,  // has been initialized
    tensorflow::Session* session_master,
    float const learning_rate_value,
    int const start_iter,  // start parallel execution, now set to 20
    int const interval_to_change_variable,  // iter interval to exchange
                                            // variable, now set to 5
    int const
        sarsa_interval,  // iter interval to run sarsa algorithm, now set to 20
    adaptive_system::sarsa_model& sm) {
    static bool is_first = true;
    for (int i = 0; i < total_iter_num_to_run; i++) {
        std::map<std::string, tensorflow::Tensor> gradients;
        std::pair<float, float> loss_results;
        PRINT_INFO;
        int level = sarsa::get_current_level();
        PRINT_INFO;
        client::compute_gradient_loss_and_quantize(session_local, level,
                                                   gradients, loss_results);
        PRINT_INFO;
        // mutex lock
        std::unique_lock<std::mutex> lk(job_update_mutex);
        std::cout << "iter_num: " << current_iter_num
                  << " loss: " << loss_results.first
                  << " entropy_loss:" << loss_results.second << std::endl;
        logging::log_to_file(0.0f, loss_results.first, loss_results.second,
                             current_iter_num, level);
        adaptive_system::apply_quantized_gradient_to_model(
            gradients, session_master, client::tuple, learning_rate_value);

        auto& id2last_iter = id_to_last_iter();
        auto& iter2gradient = adaptive_system::iter_to_gradient();
        iter2gradient[current_iter_num] = {gradients, 0};
        int last_iter;
        auto iter = id2last_iter.find(worker_id);
        if (iter == id2last_iter.end()) {
            last_iter = start_iter - 1;
        } else {
            last_iter = iter->second;
        }
        copy_gradient_between_session(
            last_iter, current_iter_num, total_worker, level,
            learning_rate_value, session_master, session_local, client::tuple,
            client::threshold_to_quantize);
        id2last_iter[worker_id] = current_iter_num;

        auto& w2it = worker_id_to_its_iter();
        w2it[worker_id]++;

        // check if it's time for exchange variables ---- not quantize
        if (w2it[worker_id] % interval_to_change_variable == 0) {
            adaptive_system::copy_variable_between_session(
                session_master, session_local, client::tuple);
        }

        // sarsa model work
        if (current_iter_num > 100) {
            sarsa::loss_history.push_back(loss_results.first);
            int const real_iter_num = current_iter_num - 100;
            if (is_first) {
                sarsa::_last_loss = loss_results.first;
                is_first = false;
            }
            if (real_iter_num % sarsa_interval == 0) {
                sarsa::adjust_rl_model(sm, level);
                sarsa::get_current_level() = level;
            }
        }
        current_iter_num++;
        lk.unlock();
    }
}

void do_work(int const total_iter_num,
             int const total_worker_num,
             int const level,
             float const learning_rate_value,
             int const start_parallel,
             std::string const tuple_path) {
    // init log
    logging::init_log(level, total_worker_num, 0, 0);
    tensorflow::Session* session_master =
        client::load_primary_model_on_master_and_init(tuple_path);
    do_work_for_one_worker(-1, start_parallel, level, session_master,
                           session_master, learning_rate_value);
    std::vector<tensorflow::Session*> session_workers;
    for (int i = 0; i < total_worker_num; i++) {
        auto session_worker =
            client::load_primary_model_on_client_and_init(session_master);
        session_workers.push_back(session_worker);
    }
    std::cout << "all init finish " << std::endl;

    std::vector<std::thread> vec_threads;
    for (int j = 0; j < total_worker_num; j++) {
        vec_threads.push_back(std::thread(
            do_work_for_one_worker, j, total_iter_num, level,
            session_workers[j], session_master, learning_rate_value));
    }
    for (int j = 0; j < total_worker_num; j++) {
        vec_threads[j].join();
    }
    std::cout << "finish training" << std::endl;
}

void do_work_v2(int const total_iter_num,
                int const total_worker_num,
                int const level,
                float const learning_rate_value,
                int const start_parallel,
                std::string const tuple_path) {
    // init log
    logging::init_log(level, total_worker_num, 0, 0);
    tensorflow::Session* session_master =
        client::load_primary_model_on_master_and_init(tuple_path);
    do_work_for_one_worker(-1, start_parallel, level, session_master,
                           session_master, learning_rate_value);
    std::vector<tensorflow::Session*> session_workers;
    for (int i = 0; i < total_worker_num; i++) {
        auto session_worker =
            client::load_primary_model_on_client_and_init(session_master);
        session_workers.push_back(session_worker);
    }
    std::cout << "all init finish " << std::endl;

    std::vector<std::thread> vec_threads;
    for (int j = 0; j < total_worker_num; j++) {
        vec_threads.push_back(
            std::thread(do_work_for_one_worker_v2, j, total_iter_num, level,
                        total_worker_num, session_workers[j], session_master,
                        learning_rate_value, start_parallel));
    }
    for (int j = 0; j < total_worker_num; j++) {
        vec_threads[j].join();
    }
    std::cout << "finish training" << std::endl;
}

void do_work_v3(int const total_iter_num,
                int const total_worker_num,
                int const level,
                float const learning_rate_value,
                int const start_parallel,
                std::string const tuple_path,
                int const interval_to_exchange_variable,
                int const sarsa_interval,
                float const sarsa_r,
                float const sarsa_eps_greedy,
                int const sarsa_start_level,
                int const sarsa_end_level,
                int const sarsa_init_level) {
    // init sarsa model
    adaptive_system::sarsa_model sm(
        "/home/cgx/git_project/adaptive-system/single_implement_sarsa_async/"
        "sarsa_continous.pb",
        sarsa_interval, sarsa_r, sarsa_eps_greedy, sarsa_start_level,
        sarsa_end_level, sarsa_init_level);
    PRINT_INFO;
    sarsa::last_state =
        tensorflow::Tensor(tensorflow::DataType::DT_FLOAT,
                           tensorflow::TensorShape({sarsa_interval}));
    float* ptr_last_state = sarsa::last_state.flat<float>().data();
    std::fill(ptr_last_state, ptr_last_state + sarsa_interval, 10.0f);

    // init log
    logging::init_log(level, total_worker_num, sarsa_interval,
                      interval_to_exchange_variable);
    tensorflow::Session* session_master =
        client::load_primary_model_on_master_and_init(tuple_path);
    do_work_for_one_worker(-1, start_parallel, level, session_master,
                           session_master, learning_rate_value);
    std::vector<tensorflow::Session*> session_workers;
    for (int i = 0; i < total_worker_num; i++) {
        auto session_worker =
            client::load_primary_model_on_client_and_init(session_master);
        session_workers.push_back(session_worker);
    }
    std::cout << "all init finish " << std::endl;

    std::vector<std::thread> vec_threads;
    for (int j = 0; j < total_worker_num; j++) {
        vec_threads.push_back(std::thread(
            do_work_for_one_worker_v3, j, total_iter_num, total_worker_num,
            session_workers[j], session_master, learning_rate_value,
            start_parallel, interval_to_exchange_variable, sarsa_interval,
            std::ref(sm)));
    }
    for (int j = 0; j < total_worker_num; j++) {
        vec_threads[j].join();
    }
    std::cout << "finish training" << std::endl;
}

}  // namespace job

int main(int argc, char** argv) {
    int const total_iter_num = atoi(argv[1]);
    int const total_worker_num = atoi(argv[2]);
    int const level = atoi(argv[3]);
    float const learning_rate_value = atof(argv[4]);
    int const start_parallel = atoi(argv[5]);
    input::batch_size = atoi(argv[6]);
    std::string const tuple_local_path = argv[7];
    int const interval_to_exchange_variable = atoi(argv[8]);
    int const sarsa_interval = atoi(argv[9]);
    float const sarsa_r = atof(argv[10]);
    float const sarsa_eps_greedy = atof(argv[11]);
    int const sarsa_start_level = atoi(argv[12]);
    int const sarsa_end_level = atoi(argv[13]);
    int const sarsa_init_level = atoi(argv[14]);
    sarsa::computing_time = atof(argv[15]);
    sarsa::one_bit_communication_time = atof(argv[16]);
    PRINT_INFO;
    input::turn_raw_tensors_to_standard_version();
    job::do_work_v3(total_iter_num, total_worker_num, level,
                    learning_rate_value, start_parallel, tuple_local_path,
                    interval_to_exchange_variable, sarsa_interval, sarsa_r,
                    sarsa_eps_greedy, sarsa_start_level, sarsa_end_level,
                    sarsa_init_level);

    return 0;
}
