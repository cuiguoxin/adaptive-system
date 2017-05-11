#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include "proto/rpc_service.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace adaptive_system {

enum QUANTIZATION_TYPE {
  TWO_BIT = 2,
  FOUR_BIT = 4,
  EIGHT_BIT = 8,
  SIXTEEN_BIT = 16,
  NO_QUANTIZATION = 32
};
void quantize(const QUANTIZATION_TYPE type,
              tensorflow::Tensor const& raw_tensor, float const max_value,
              float const min_value, Gradient& grad);
void dequantize(const QUANTIZATION_TYPE type, Gradient& grad,
                tensorflow::Tensor& raw_tensor)
}
