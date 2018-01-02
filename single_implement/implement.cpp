#include <algorithm>
#include <chrono>
#include <cmath>
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

#include "single_implement/accuracy.h"

namespace input {
using namespace tensorflow;

unsigned int index_current = 0;
int const batch_size = 32;
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
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(raw_tensors.begin(), raw_tensors.end(),
                 std::default_random_engine(seed));
}
}  // namespace

void turn_raw_tensors_to_standard_version(
    const std::string& binary_file_prefix =
        "/home/cgx/git_project/adaptive-system/resources/cifar-10-batches-bin/"
        "data_batch_",
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
Session* session;
std::string batch_placeholder_name;
std::string label_placeholder_name;
const int threshold_to_quantize = 105000;

namespace {

float get_norm(tensorflow::Tensor& tensor) {
    int const size = tensor.NumElements();
    float const* tensor_ptr = tensor.flat<float>().data();
    float sum = 0;
    for (int i = 0; i < size; i++) {
        float const current_value = tensor_ptr[i];
        sum += current_value * current_value;
    }
    return pow(sum / size, 0.5f);
}

std::pair<float, float> compute_gradient_and_loss(
    std::vector<std::pair<std::string, tensorflow::Tensor>> feeds,
    std::map<std::string, tensorflow::Tensor>& gradients) {
    std::vector<std::string> fetch;
    std::string loss_name = tuple.loss_name();
    fetch.push_back(loss_name);
    auto cross_entropy_loss_name = tuple.cross_entropy_loss_name();
    fetch.push_back(cross_entropy_loss_name);
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
    tensorflow::Status tf_status = session->Run(feeds, fetch, {}, &outputs);
    if (!tf_status.ok()) {
        PRINT_ERROR_MESSAGE(tf_status.error_message());
        std::terminate();
    }
    tensorflow::Tensor& loss_tensor = outputs[0];
    float* loss_ptr = loss_tensor.flat<float>().data();
    float loss_ret = loss_ptr[0];
    auto& cross_entropy_loss_tensor = outputs[1];
    float* cross_entropy_loss_ptr =
        cross_entropy_loss_tensor.flat<float>().data();
    float cross_entropy_loss = cross_entropy_loss_ptr[0];
    outputs.erase(outputs.begin());
    outputs.erase(outputs.begin());

    size_t size = outputs.size();
    for (size_t i = 0; i < size; i++) {
        gradients.insert(std::pair<std::string, tensorflow::Tensor>(
            variable_names_in_order[i], outputs[i]));
    }
    return {loss_ret, cross_entropy_loss};
}
}  // namespace

void load_primary_model_and_init() {
    const std::string tuple_local_path =
        "/home/cgx/git_project/adaptive-system/input/cifar10/"
        "tuple_gradient_descent.pb";
    session = tensorflow::NewSession(tensorflow::SessionOptions());
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
    Status tf_status = session->Create(graph_def);
    if (!tf_status.ok()) {
        std::cout << "create graph has failed in line " << __LINE__
                  << " in file " << __FILE__ << std::endl;
        std::terminate();
    }

    // init parameters
    std::string init_name = tuple.init_name();
    std::cout << init_name << std::endl;
    tf_status = session->Run({}, {}, {init_name}, nullptr);
    if (!tf_status.ok()) {
        std::cout << "running init has  failed in line " << __LINE__
                  << " in file " << __FILE__ << std::endl;
        std::terminate();
    }
    // init some names
    batch_placeholder_name = tuple.batch_placeholder_name();
    label_placeholder_name = tuple.label_placeholder_name();
}

void compute_gradient_loss_and_quantize(
    const int level,
    std::map<std::string, tensorflow::Tensor>& map_gradients,
    std::pair<float, float>& loss) {
    PRINT_INFO;
    std::pair<tensorflow::Tensor, tensorflow::Tensor> feeds =
        input::get_next_batch();
    PRINT_INFO;
    loss = compute_gradient_and_loss({{batch_placeholder_name, feeds.first},
                                      {label_placeholder_name, feeds.second}},
                                     map_gradients);

    // get norm
    float norm = 0;
    for(auto pair : map_gradients) {
        auto& name = pair.first;
        auto & tensor  = pair.second;
        int const size = tensor.NumElements();
        if(name.find("local4") != std::string::npos && size > threshold_to_quantize) {
            norm = get_norm(tensor);
        }
    }
    PRINT_INFO;
    NamedGradientsAccordingColumn named_gradients_send;
    quantize_gradients_according_column(map_gradients, &named_gradients_send,
                                        level, threshold_to_quantize);
    map_gradients.clear();
    dequantize_gradients_according_column(named_gradients_send, map_gradients);
}

namespace log {

std::ofstream file_loss_stream;

void init_log(int const total_worker_num,
              int const pre_level,
              int const split_point,
              int const post_level,
              float const init_lr,
              int const start_iter_num) {
    // init log
    auto now = std::chrono::system_clock::now();
    auto init_time_t = std::chrono::system_clock::to_time_t(now);
    std::string label = std::to_string(init_time_t);
    std::string store_loss_file_path =
        "baseline_" + label +
        "_number_of_workers:" + std::to_string(total_worker_num) + "_" +
        std::to_string(pre_level) + "-" + std::to_string(split_point) + "-" +
        std::to_string(post_level) + "_initLr-" + std::to_string(init_lr) +
        "_startIterNum-" + std::to_string(start_iter_num);
    file_loss_stream.open("loss_result/" + store_loss_file_path);
    init(store_loss_file_path);
}

inline void log(float const time,
                float const total_loss,
                float const cross_entropy_loss,
                int const current_iter,
                int const current_level) {
    file_loss_stream << std::to_string(time)
                     << ":: iter num ::" << std::to_string(current_iter)
                     << ":: loss is ::" << total_loss << "::" << current_level
                     << ":: cross_entropy_loss is ::" << cross_entropy_loss
                     << "\n";
    file_loss_stream.flush();
}

}  // namespace log

void do_work(int const total_iter_num,
             int const total_worker_num,
             int const pre_level,
             int const split_point,
             int const post_level,
             float const init_lr,
             int const start_iter_num,
             int const predict_interval) {
    log::init_log(total_worker_num, pre_level, split_point, post_level, init_lr,
                  start_iter_num);
    load_primary_model_and_init();
    int level = 0;
    float lr = init_lr;
    std::vector<int> quantize_levels;
    for (int i = 0; i < total_iter_num; i++) {
        if (i < start_iter_num) {
            level = 2;
        } else if (i < split_point) {
            level = pre_level;
            lr = 0.2;
        } else {
            level = post_level;
            lr = 0.2;
        }
        std::vector<std::map<std::string, tensorflow::Tensor>> vec_grads;
        std::vector<std::thread> vec_threads;
        std::vector<std::pair<float, float>> vec_losses;
        vec_losses.resize(total_worker_num);
        vec_grads.resize(total_worker_num);
        for (int j = 0; j < total_worker_num; j++) {
            vec_threads.push_back(
                std::thread(compute_gradient_loss_and_quantize, level,
                            std::ref(vec_grads[j]), std::ref(vec_losses[j])));
        }
        for (int j = 0; j < total_worker_num; j++) {
            vec_threads[j].join();
        }
        PRINT_INFO;
        // finished gradient computing
        std::map<std::string, tensorflow::Tensor> merged_gradient;
        aggregate_gradients(vec_grads, merged_gradient);
        PRINT_INFO;
        average_gradients(total_worker_num, merged_gradient);
        NamedGradientsAccordingColumn store_named_gradient;
        PRINT_INFO;
        quantize_gradients_according_column(merged_gradient,
                                            &store_named_gradient, level,
                                            threshold_to_quantize);
        PRINT_INFO;
        apply_quantized_gradient_to_model(store_named_gradient, session, tuple,
                                          lr);
        // log
        std::vector<float> total_losses;
        std::vector<float> cross_entropy_losses;
        for (auto pair : vec_losses) {
            total_losses.push_back(pair.first);
            cross_entropy_losses.push_back(pair.second);
        }
        float const total_average =
            std::accumulate(total_losses.begin(), total_losses.end(), 0.0f) /
            total_worker_num;
        float const cross_entropy_average =
            std::accumulate(cross_entropy_losses.begin(),
                            cross_entropy_losses.end(), 0.0f) /
            total_worker_num;
        log::log(0, total_average, cross_entropy_average, i, level);
        std::cout << "iter num is : " << i << " loss is : " << total_average
                  << " cross entropy loss is " << cross_entropy_average
                  << std::endl;
        quantize_levels.push_back(level);
        if (i % predict_interval == 0) {
            predict(client::session, i, quantize_levels);
            quantize_levels.clear();
        }
    }
}

}  // namespace client

int main(int argc, char** argv) {
    int const total_iter_num = atoi(argv[1]);
    int const total_worker_num = atoi(argv[2]);
    int const pre_level = atoi(argv[3]);
    int const split_point = atoi(argv[4]);
    int const post_level = atoi(argv[5]);
    float const init_lr = atof(argv[6]);
    int const start_iter_num = atoi(argv[7]);
    int const predict_interval = atoi(argv[8]);

    input::turn_raw_tensors_to_standard_version();
    client::do_work(total_iter_num, total_worker_num, pre_level, split_point,
                    post_level, init_lr, start_iter_num, predict_interval);

    return 0;
}
