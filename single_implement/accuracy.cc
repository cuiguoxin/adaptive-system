#include "single_implement/accuracy.h"

namespace cifar10 {}
tensorflow::Tensor _labels, _images;

void init_image_label() {
    const std::string &binary_file_path = "",
        const std::string &preprocess_graph_path = "";
    cifar10::turn_raw_tensors_to_standard_version(binary_file_path,
                                                  preprocess_graph_path);
    auto image_and_label = cifar10::get_next_batch(10000);
    _images = image_and_label.first;
    _labels = image_and_label.second;
}

void predict_periodically(std::string const& batch_placeholder_name,
                          std::string const& label_placeholder_name,
                          std::string const& loss_name,
                          std::string const tuple_predict_path, 
                          tensorflow::Session* session) {
    std::string image_name = batch_placeholder_name;
    std::string label_name = label_placeholder_name;
    std::string loss_name_copy = loss_name;
    tensorflow::Session* session = init_predict_session(tuple_predict_path);
    while (true) {
        // first assign
        assign_predict_variables(_session, session, _tuple);
        // then predict
        std::vector<tensorflow::Tensor> loss_vec;
        tensorflow::Status status =
            session->Run({{image_name, _images}, {label_name, _labels}},
                         {loss_name_copy}, {}, &loss_vec);
        if (!status.ok()) {
            PRINT_ERROR_MESSAGE(status.error_message());
            std::terminate();
        }
        auto loss_tensor = loss_vec[0];
        float* loss = loss_tensor.flat<float>().data();
        _file_predict_stream << std::to_string(_current_iter_number)
                             << " :: " << std::to_string(*loss) << "\n";
        _file_predict_stream.flush();
    }
}

tensorflow::Session* init_predict_session(std::string tuple_predict_path) {
    tensorflow::Session* session =
        tensorflow::NewSession(tensorflow::SessionOptions());
    std::fstream input(tuple_predict_path, std::ios::in | std::ios::binary);
    Tuple tuple_predict;
    if (!input) {
        std::cout << tuple_predict_path
                  << ": File not found.  Creating a new file." << std::endl;
    } else if (!tuple_predict.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse tuple." << std::endl;
        std::terminate();
    }

    tensorflow::GraphDef graph_def = tuple_predict.graph();
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

void start_accuracy() {
    std::thread predict_thread(&RPCServiceImpl::predict_periodically, this,
                               std::ref(_image_placeholder_name),
                               std::ref(_label_placeholder_name),
                               std::ref(_loss_name), tuple_predict_path);
    predict_thread.detach();
}
