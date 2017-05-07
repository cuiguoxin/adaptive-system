#include "quantization/util/algorithm.h"

namespace adaptive_system {

    namespace {
        void quantize_less_8_bits(const QUANTIZATION_TYPE type,
                                float* raw_data, 
                                const float max_value,
                                const float min_value,
                                const size_t raw_data_length,
                                tensorflow::uint8 ** quantized_data,
                                size_t& quantized_data_length // the number of bytes to output
                                ) {
            const int q_type = static_cast<int>(type); //for example 2
            const int scope = std::pow(2, q_type);
            const float multiplizer = scope / (max_value - min_value);
            std::for_each(raw_data, raw_data + raw_data_length, [multiplizer, min_value](float& ref){
                                                                             ref = (ref - min_value) * multiplizer; 
                                                                             });
            const int length_per_iter = 8 / q_type; //for example 4
            quantized_data_length = static_cast<size_t>(std::ceil(raw_data_length / 
                                                                  static_cast<float>(length_per_iter)));
            std::cout << "quantized_data_length is " << quantized_data_length << std::endl;
            tensorflow::uint8* output = new tensorflow::uint8[quantized_data_length];
            for (int i = 0; i < quantized_data_length; i++) {
                output[i] = 0;
                for (int j = 0; j < length_per_iter; j++) {
                    const int index_for_raw = length_per_iter * i + j;
                    if (index_for_raw >= raw_data_length) {
                        break;
                    }
                    int value_raw = raw_data[index_for_raw];
                    output[i] = output[i] | (value_raw << (q_type * j));
                }
            }
            *quantized_data = output;
        } 
        void dequantize_less_8_bits(const QUANTIZATON_TYPE type,
                    const tensorflow::uint8* quantized_data,
                    const size_t quantized_data_length,
                    const size_t raw_data_length,
                    const float max_value,
                    const float min_value,
                    float* raw_data
                    ) {
            static const tensorflow::uint8 mask_2_bits = 3, mask_4_bits = 15, mask_8_bits = 255;
            const int q_type = static_cast<int>(type); //for example 2
            const int scope = std::pow(2, q_type);
            const int length_per_iter = 8 / q_type; //for example 4
            const float multiplier = (max_value - min_value) / scope;
            int i = 0; 
            std::function<void(float&)> func = [=, &i](float& ref) {
                const int index_for_q_data = i / lenth_per_iter;
                const int index_in_iter = i - index_for_q_data * length_per_iter;
                tensorflow::uint8 q_data = quantized_data[index_for_q_data];
                const int move_right = q_type * index_in_iter;
                q_data = q_data >> move_right;
                switch (q_type) {
                    case 2: q_data &= mask_2_bits;
                            break;
                    case 4: q_data &= mask_4_bits;
                            break;
                    case 8: q_data &= mask_8_bits;
                            break;
                }
                ref = q_data * multiplier + min_value * 1.5;
            };
            std::for_each(raw_data, raw_data + raw_data_length, func);
        }  
    }
    
    void quantize(const QUANTIZATION_TYPE type,
                  float* raw_data, 
                  const float max_value,
                  const float min_value,
                  const size_t raw_data_length,
                  tensorflow::uint8 ** quantized_data,
                  size_t& quantized_data_length
                  ) {
         quantize_less_8_bits(type, raw_data, max_value, min_value, raw_data_length, quantized_data, quantized_data_length);          
    }
    //raw data must be allocated outside this function
    void dequantize(const QUANTIZATON_TYPE type,
                    const tensorflow::uint8* quantized_data,
                    const size_t quantized_data_length,
                    const size_t raw_data_length,
                    const float max_value,
                    const float min_value,
                    float* raw_data
                    ) {
        dequantize_less_8_bits(type, quantized_data, quantized_data_length, raw_data_length, max_value, min_value, raw_data);
    }
}

