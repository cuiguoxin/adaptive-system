#include "input/cifar10/input.h"

using namespace tensorflow;
namespace cifar10 {

namespace {
int index_current = 0;
std::vector<Tensor> raw_tensors;
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
      input_stream.read(raw_tensor_ptr, record_size);
      raw_tensors.push_back(raw_tensor);
    }
  }
  input_stream.close();
  // shuffle the vector raw_tensors
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(raw_tensors.begin(), raw_tensors.end(),
               std::default_random_engine(seed));
}
}

void turn_raw_tensors_to_standard_version(const std::string& binary_file_path,
                                          const std::string& graph_path) {
  Session* session = load_graph_and_create_session(graph_path);
  read_raw_tensors_from_file(binary_file_path);
}
}