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
}
