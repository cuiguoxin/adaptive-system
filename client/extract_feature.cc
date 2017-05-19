#include "client/extract_feature.h"

namespace adpative_system {
namespace {

void mean(tensorflow::Tensor& const tensor, float& result) {
  size_t size = tensor.NumElements();
  float const* tensor_ptr = tensor.flat<float>().data();
  float sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += tensor_ptr[i];
  }
  result = sum / size;
}
void max(tensorflow::Tensor& const tensor, float& result) {
  size_t size = tensor.NumElements();
  float const* tensor_ptr = tensor.flat<float>().data();
  result = tensor_ptr[0];
  for (size_t i = 1; i < size; i++) {
    if (result < tensor_ptr[i]) result = tensor_ptr[i];
  }
}
void min(tensorflow::Tensor& const tensor, float& result) {
  size_t size = tensor.NumElements();
  float const* tensor_ptr = tensor.flat<float>().data();
  result = tensor_ptr[0];
  for (size_t i = 1; i < size; i++) {
    if (result > tensor_ptr[i]) result = tensor_ptr[i];
  }
}
void deviation(tensorflow::Tensor& const tensor, float& result) {
  size_t size = tensor.NumElements();
  float const* tensor_ptr = tensor.flat<float>().data();
  float sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += tensor_ptr[i];
  }
  float average = sum / size;
  float deviation_sum = 0;
  for (size_t i = 0; i < size; i++) {
    deviation_sum += std::pow(tensor_ptr[i] - average, 2.0);
  }
  deviation_sum = deviation_sum / size;
  result = std::pow(deviation_sum, 0.5);
}
void abs_sum(tensorflow::Tensor& const tensor, float& result) {
  size_t size = tensor.NumElements();
  float const* tensor_ptr = tensor.flat<float>().data();
  result = 0;
  for (size_t i = 0; i < size; i++) {
    if (tensor_ptr[i] > 0)
      result += tensor_ptr[i];
    else
      result -= tensor_ptr[i];
  }
}
void median(tensorflow::Tensor& const tensor, float& result) {
  size_t size = tensor.NumElements();
  float const* tensor_ptr = tensor.flat<float>().data();
  float* float_new = new float[size];
  std::copy(tensor_ptr, tensor_ptr + size, float_new);
  std::nth_element(float_new, float_new + size / 2, float_new + size);
  result = float_new[size / 2];
  delete[] float_new;
}
void norm(tensorflow::Tensor& const tensor, float& result) {
  size_t size = tensor.NumElements();
  float const* tensor_ptr = tensor.flat<float>().data();
  result = 0;
  for (size_t i = 0; i < size; i++) {
    result += std::pow(tensor_ptr[i], 2.0);
  }
}
}

tensorflow::Tensor get_feature(tensorflow::Tensor const& tensor) {
  tensorflow::Tensor ret_tensor =
      Tensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({7}));
  float* ret_tensor_ptr = ret_tensor.flat<float>().data();
  std::thread mean_thread(mean, std::ref(tensor), std::ref(ret_tensor_ptr[0]));
  std::thread min_thread(min, std::ref(tensor), std::ref(ret_tensor_ptr[1]));
  std::thread max_thread(max, std::ref(tensor), std::ref(ret_tensor_ptr[2]));
  std::thread deviation_thread(deviation, std::ref(tensor),
                               std::ref(ret_tensor_ptr[3]));
  std::thread abs_sum_thread(abs_sum, std::ref(tensor),
                             std::ref(ret_tensor_ptr[4]));
  std::thread median_thread(median, std::ref(tensor),
                            std::ref(ret_tensor_ptr[5]));
  std::thread norm_thread(norm, std::ref(tensor), std::ref(ret_tensor_ptr[6]));
  mean_thread.join();
  min_thread.join();
  max_thread.join();
  deviation_thread.join();
  abs_sum_thread.join();
  median_thread.join();
  norm_thread.join();

  return ret_tensor;
}
}