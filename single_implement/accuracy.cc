#include "single_implement/accuracy.h"

using namespace tensorflow;
namespace cifar10 {}
tensorflow::Tensor labels, images;
std::ofstream accuracy_stream;
tensorflow::Session *session_to;
adaptive_system::Tuple predict_tuple;
std::string image_name, label_name, loss_name, accuracy_name,
    cross_entropy_loss_name;

void init_preprocess() {
    // init preprocess graph
    GraphDef graph_def;
    std::string graph_path =
        "/home/cgx/git_project/adaptive-system/input/cifar10/preprocess.pb";
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
    // read raw tensor from file
    const int record_size = 3073;
    const int label_size = 1;
    const int image_size = 3072;
    const int batch_size = 10000;
    std::ifstream input_stream(
        "/home/cgx/git_project/adaptive-system/resources/cifar-10-batches-bin/"
        "test_batch.bin",
        std::ios::binary);
    TensorShape raw_tensor_shape({record_size});
    std::vector<tensorflow::Tensor> raw_tensors;
    if (input_stream.is_open()) {
        for (int j = 0; j < batch_size; j++) {
            Tensor raw_tensor(DataType::DT_UINT8, raw_tensor_shape);
            uint8* raw_tensor_ptr = raw_tensor.flat<uint8>().data();
            input_stream.read(reinterpret_cast<char*>(raw_tensor_ptr),
                              record_size);
            raw_tensors.push_back(raw_tensor);
        }
    }
    input_stream.close();

    // preprocess the images
    std::vector<tensorflow::Tensor> standard_images, standard_labels;
    for (int i = 0; i < batch_size; i++) {
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
    TensorShape images_batch_shape({batch_size, 28, 28, 3}),
        labels_batch_shape({batch_size});
    Tensor images_batch(DataType::DT_FLOAT, images_batch_shape),
        labels_batch(DataType::DT_INT32, labels_batch_shape);
    float* images_batch_ptr = images_batch.flat<float>().data();
    int* label_batch_ptr = labels_batch.flat<int>().data();
    std::unique_lock<std::mutex> lk(mu);
    for (int i = 0; i < batch_size; i++) {
        Tensor& image_current = standard_images[i];
        float* image_current_ptr = image_current.flat<float>().data();
        std::copy(image_current_ptr, image_current_ptr + standard_images_size,
                  images_batch_ptr + i * standard_images_size);
        Tensor& label_current = standard_labels[i];
        int* label_current_ptr = label_current.flat<int>().data();
        label_batch_ptr[i] = *label_current_ptr;
    }
    images = images_batch;
    labels = labels_batch;
}

void init(std::string const log_file_name) {
    // init image and label
    init_preprocess();
    // init log
    accuracy_stream.open("accuracy/" + log_file_name);
    // init session and tuple
    std::string const tuple_predict_path =
        "/home/cgx/git_project/adaptive-system/input/cifar10/"
        "tuple_predict_accuracy.pb";
    session_to = init_predict_session(tuple_predict_path);
    
    // init other string labels
    image_name = predict_tuple.batch_placeholder_name();
    label_name = predict_tuple.label_placeholder_name();
    loss_name = predict_tuple.loss_name();
    accuracy_name = predict_tuple.accuracy_name();
    cross_entropy_loss_name = predict_tuple.cross_entropy_loss_name();
}

void predict(tensorflow::Session* session_from,
             int const current_iter_num,
             std::vector<int> const& quantize_levels) {
    assign_predict_variables(session_from, session_to, _tuple);
    // then predict
    std::vector<tensorflow::Tensor> loss_vec;
    tensorflow::Status status = session_to->Run(
        {{image_name, _images}, {label_name, _labels}},
        {loss_name, accuracy_name, cross_entropy_loss_name}, {}, &loss_vec);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
    auto loss_tensor = loss_vec[0];
    float* loss = loss_tensor.flat<float>().data();
    auto predict_accuracy_tensor = loss_vec[1];
    bool* predict_accuracy_ptr = predict_accuracy_tensor.flat<bool>().data();
    auto cross_entropy_loss_tensor = loss_vec[2];
    float* cross_entropy_loss_ptr =
        cross_entropy_loss_tensor.flat<float>().data();
    std::string str_quantize_levels;
    for (auto level : quantize_levels) {
        str_quantize_levels += std::to_string(level) + " ";
    }
    float right = 0;
    for (int i = 0; i < 10000; i++) {
        if (predict_accuracy_ptr[i]) {
            right = right + 1;
        }
    }
    float final_accuracy = right / 10000;
    accuracy_stream << "iter num:: " << std::to_string(current_iter_number)
                    << ":: total_loss :: " << std::to_string(*loss)
                    << ":: entropy_loss :: "
                    << std::to_string(*cross_entropy_loss_ptr)
                    << ":: predict_accuracy ::"
                    << std::to_string(final_accuracy)
                    << ":: quantize_levels ::" << str_quantize_levels << "\n";
    accuracy_stream.flush();
}

tensorflow::Session* init_predict_session(std::string tuple_predict_path) {
    tensorflow::Session* session =
        tensorflow::NewSession(tensorflow::SessionOptions());
    std::fstream input(tuple_predict_path, std::ios::in | std::ios::binary);
    if (!input) {
        std::cout << tuple_predict_path
                  << ": File not found.  Creating a new file." << std::endl;
    } else if (!predict_tuple.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse tuple." << std::endl;
        std::terminate();
    }

    tensorflow::GraphDef graph_def = predict_tuple.graph();
    tensorflow::Status tf_status = session->Create(graph_def);
    if (!tf_status.ok()) {
        std::cout << tf_status.error_message() << std::endl;
        std::terminate();
    }
    return session;
}

void assign_predict_variables(tensorflow::Session* from,
                              tensorflow::Session* to,
                              Tuple const& tuple) {
    std::vector<std::string> variable_names;
    for (auto& pair : tuple.map_names()) {
        variable_names.push_back(pair.first);
    }
    std::vector<tensorflow::Tensor> values;
    tensorflow::Status status = from->Run({}, variable_names, {}, &values);
    if (!status.ok()) {
        PRINT_ERROR_MESSAGE(status.error_message());
        std::terminate();
    }
    google::protobuf::Map<std::string, Names> const& map_names =
        tuple.map_names();
    std::vector<std::string> assign_names;
    std::vector<std::pair<std::string, tensorflow::Tensor>> feeds;
    int i = 0;
    for (std::string const& variable_name : variable_names) {
        auto& names = map_names.find(variable_name)->second;
        assign_names.push_back(names.assign_name());
        feeds.push_back(std::pair<std::string, tensorflow::Tensor>(
            names.placeholder_assign_name(), values[i]));
        i++;
    }
    tensorflow::Status tf_status = to->Run(feeds, {}, assign_names, nullptr);
    if (!tf_status.ok()) {
        PRINT_ERROR_MESSAGE(tf_status.error_message());
        std::terminate();
    }
}

