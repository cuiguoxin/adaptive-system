#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>
#include "tensorflow/core/platform/types.h"

namespace adaptive_system {

    enum QUANTIZATION_TYPE {
        TWO_BIT = 2,
        FOUR_BIT = 4,
        EIGHT_BIT = 8,
        SIXTEEN_BIT = 16,
        NO_QUANTIZATION = 32
    };

    //may change the raw_data
    //quantized_data must be allocated in this function, and quantized_data_length means the number of bytes in quantized_data
    void quantize(const QUANTIZATION_TYPE type,
                  float* raw_data, 
                  const float max_value,
                  const float min_value,
                  const size_t raw_data_length,
                  tensorflow::uint8 ** quantized_data,
                  size_t& quantized_data_length
                  );
    //raw data must be allocated outside this function
    void dequantize(const QUANTIZATION_TYPE type,
                    const tensorflow::uint8* quantized_data,
                    const size_t quantized_data_length,
                    const size_t raw_data_length,
                    const float max_value,
                    const float min_value,
                    float* raw_data
                    );

}
