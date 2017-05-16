#include "input/cifar10/input.h"

using namespace tensorflow;
namespace cifar10 {

namespace {
unsigned int index_current = 0;
int const batch_size = 64;
std::vector<Tensor> raw_tensors, standard_images, standard_labels;
const int record_size = 3073;
const int label_size = 1;
const int image_size = 3072;

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

void read_raw_tensors_from_file(const std::string& binary_file_path) {
  std::ifstream input_stream(binary_file_path, std::ios::binary);
  TensorShape raw_tensor_shape({record_size});
  if (input_stream.is_open()) {
    for (int i = 0; i < 10000; i++) {
      Tensor raw_tensor(DataType::DT_UINT8, raw_tensor_shape);
      uint8* raw_tensor_ptr = raw_tensor.flat<uint8>().data();
      input_stream.read(reinterpret_cast<char*>(raw_tensor_ptr), record_size);
      raw_tensors.push_back(raw_tensor);
    }
  }
  input_stream.close();
  // shuffle the vector raw_tensors
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(raw_tensors.begin(), raw_tensors.end(),
               std::default_random_engine(seed));
  for (int i = 0; i < 10; i++) {
    Tensor ten = raw_tensors[i];
    uint8* ten_ptr = ten.flat<uint8>().data();
    for (int j = 0; j < 3073; i++) {
      std::cout << ten_ptr[j] << " ";
    }
    std::cout << std::endl;
  }
}
}

void turn_raw_tensors_to_standard_version(const std::string& binary_file_path,
                                          const std::string& graph_path) {
  Session* session = load_graph_and_create_session(graph_path);
  read_raw_tensors_from_file(binary_file_path);
  for (int i = 0; i < 10000; i++) {
    Tensor raw_tensor = raw_tensors[i];
    std::vector<Tensor> image_and_label;
    Status status = session->Run({{"raw_tensor", raw_tensor}}, {"div", "label"},
                                 {}, &image_and_label);
    if (!status.ok()) {
      std::cout << "failed in line " << __LINE__ << " in file " << __FILE__
                << " " << status.error_message() << std::endl;
      std::terminate();
    }
    standard_images.push_back(image_and_label[0]);
    standard_labels.push_back(image_and_label[1]);
  }
  raw_tensors.clear();
}

std::pair<Tensor, Tensor> get_next_batch() {
  int standard_images_size = 3 * 28 * 28;
  TensorShape images_batch_shape({batch_size, 28, 28, 3}),
      labels_batch_shape({batch_size});
  Tensor images_batch(DataType::DT_FLOAT, images_batch_shape),
      labels_batch(DataType::DT_INT32, labels_batch_shape);
  float* images_batch_ptr = images_batch.flat<float>().data();
  int* label_batch_ptr = labels_batch.flat<int>().data();
  for (int i = 0; i < batch_size; i++) {
    int real_index = index_current % 10000;
    Tensor& image_current = standard_images[real_index];
    float* image_current_ptr = image_current.flat<float>().data();
    std::copy(image_current_ptr, image_current_ptr + standard_images_size,
              images_batch_ptr + i * standard_images_size);
    Tensor& label_current = standard_labels[real_index];
    int* label_current_ptr = label_current.flat<int>().data();
    label_batch_ptr[i] = *label_current_ptr;
    index_current++;
  }
  return std::pair<Tensor, Tensor>(images_batch, labels_batch);
}
}