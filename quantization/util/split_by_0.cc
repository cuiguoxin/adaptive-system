#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <utility>
#include "quantization/util/qsgd.h"

namespace adaptive_system {

namespace split_by_0 {

namespace {

float get_abs(float value) {
    if (value > 0) {
        return value;
    } else {
        return -value;
    }
}

void get_abs_max(tensorflow::Tensor const& tensor, float& max) {
    float const* tensor_ptr = tensor.flat<float>().data();
    size_t size = tensor.NumElements();
    max = get_abs(tensor_ptr[0]);
    std::for_each(tensor_ptr, tensor_ptr + size, [&max](float const current) {
        float abs_current = get_abs(current);
        if (max < abs_current) {
            max = abs_current;
            return;
        }
    });
}

// 0 <= start <= end <= 7, and assume that value is less than 2^(end - start)
void set_byte(uint8_t* byte,
              size_t const start,
              size_t const end,
              uint8_t const value) {
    uint8_t left_offset = 7 - end;
    uint8_t value_left_move = value << left_offset;
    *byte |= value_left_move;
}

uint32_t read_byte(uint8_t const* byte, size_t const start, size_t const end) {
    uint8_t left_moved = (*byte) << start;
    uint8_t right_moved = left_moved >> start >> (7 - end);
    return right_moved;
}

uint32_t read_uint32(uint32_t const value,
                     size_t const start,
                     size_t const end) {
    uint32_t right_moved = value >> (31 - end);
    uint32_t right_moved_again = right_moved >> (end - start + 1);
    uint32_t left_move = right_moved_again << (end - start + 1);
    return right_moved - left_move;
}

void set_value(uint8_t* arr,
               size_t const start,
               size_t const length,
               uint32_t const value) {
    size_t const index_in_array_begin = start / 8;
    size_t const index_in_byte_begin = start - index_in_array_begin * 8;
    size_t const end = start + length - 1;  // must minus 1
    size_t const index_in_array_end = end / 8;
    size_t const index_in_byte_end = end - index_in_array_end * 8;
    if (index_in_array_begin == index_in_array_end) {
        set_byte(arr + index_in_array_begin, index_in_byte_begin,
                 index_in_byte_end, value);
    } else {
        size_t iter_begin = 32 - length;
        size_t iter_end = iter_begin + (7 - index_in_byte_begin);
        uint32_t value_to_set = read_uint32(value, iter_begin, iter_end);
        set_byte(arr + index_in_array_begin, index_in_byte_begin, 7,
                 value_to_set);
        for (size_t i = index_in_array_begin + 1; i < index_in_array_end; i++) {
            arr[i] = 0;
            iter_begin = iter_end + 1;
            iter_end = iter_end + 8;
            value_to_set = read_uint32(value, iter_begin, iter_end);
            set_byte(arr + i, 0, 7, value_to_set);
        }
        iter_begin = iter_end + 1;
        iter_end = 31;
        value_to_set = read_uint32(value, iter_begin, iter_end);
        set_byte(arr + index_in_array_end, 0, index_in_byte_end, value_to_set);
    }
}

uint32_t read_value(uint8_t const* arr,
                    size_t const start,
                    size_t const length) {
    size_t const index_in_array_begin = start / 8;
    size_t const index_in_byte_begin = start - index_in_array_begin * 8;
    size_t const end = start + length - 1;  // must minus 1
    size_t const index_in_array_end = end / 8;
    size_t const index_in_byte_end = end - index_in_array_end * 8;
    if (index_in_array_begin == index_in_array_end) {
        return read_byte(arr + index_in_array_begin, index_in_byte_begin,
                         index_in_byte_end);
    } else {
        size_t iter_begin = index_in_byte_begin, iter_end = 7;
        uint32_t result =
            read_byte(arr + index_in_array_begin, iter_begin, iter_end);
        for (size_t i = index_in_array_begin + 1; i < index_in_array_end; i++) {
            uint8_t value_byte = arr[i];
            result = (result << 8) + value_byte;
        }
        uint32_t last =
            read_byte(arr + index_in_array_end, 0, index_in_byte_end);
        result = (result << (index_in_byte_end + 1)) + last;
        return result;
    }
}
}  // namespace

void quantize_gradient_according_column(uint32_t const level,
                                        tensorflow::Tensor const& tensor,
                                        GradientAccordingColumn& gradient) {
    unsigned int now = time(NULL);
    PRINT_INFO;
    gradient.set_is_qsgd(true);
    auto dims = tensor.dims();
    if (dims != 2) {
        PRINT_ERROR_MESSAGE("dims is not 2");
        std::terminate();
    }
    int const dim1 = tensor.dim_size(0);
    int const dim2 = tensor.dim_size(1);
    std::vector<float> max_vector, min_vector;
    max_vector.resize(dim2, std::numeric_limits<float>::min());
    min_vector.resize(dim2, std::numeric_limits<float>::max());
    float const* tensor_ptr = tensor.flat<float>().data();
    // get the max and min values
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            float current_value = tensor_ptr[dim2 * j + i];
            if (max_vector[i] < current_value) {
                max_vector[i] = current_value;
            }
            if (min_vector[i] > current_value) {
                min_vector[i] = current_value;
            }
        }
    }
    // int size_signs = std::ceil(dim1 * dim2 / 8.0f);
    // uint8_t* signs = new uint8_t[size_signs]();
    unsigned long long const scope = ((long long)1) << level;
    float const eps = 0.000001;
    // quantize each column
    // int sign_begin = 0;
    PRINT_INFO;
    for (int i = 0; i < dim2; i++) {
        float* col_ptr = new float[dim1]();
        for (int j = 0; j < dim1; j++) {
            col_ptr[j] = tensor_ptr[dim2 * j + i];
        }
        size_t quantized_size =
            std::ceil(((float)dim1) * level / 8);  // number of byte
        uint8_t* quantized_data = new uint8_t[quantized_size]();
        float const max = max_vector[i];
        float const min = min_vector[i];
        float const positive_multiplier = scope / 2 / (max + eps);
        float const negative_multiplier = scope / 2 / (-min + eps);
        size_t begin = 0;
        for (size_t j = 0; j < dim1; j++) {
            float const current_value = col_ptr[j];
            int value = -1;
            if (current_value > 0) {
                value = positive_multiplier * current_value + scope / 2 - 1;
            } else {
                value = negative_multiplier * (-current_value);
            }
            // float const r = 0.5;
            // int const value = (r > diff) ? value_int : value_int + 1;
            set_value(quantized_data, begin, level, value);
            begin += level;
            // int const sign = (col_ptr[j] > 0) ? 1 : 0;
            // set_value(signs, sign_begin, 1, sign);
            // sign_begin++;
        }
        gradient.add_maxes(max);
        gradient.add_mins(min);
        gradient.add_quantized_columns(quantized_data, quantized_size);

        delete[] quantized_data;
        delete[] col_ptr;
    }
    PRINT_INFO;
    // gradient.set_signs(signs, size_signs);
    // delete[] signs;

    // assign to gradient
    gradient.set_dim1(dim1);
    gradient.set_dim2(dim2);
    gradient.set_quantization_level(level);
}

void dequantize_gradient_according_column(
    GradientAccordingColumn const& gradient,
    tensorflow::Tensor& tensor) {
    PRINT_INFO;
    int const level = gradient.quantization_level();
    unsigned long long const scope = ((long long)1) << level;
    int const dim1 = gradient.dim1();
    int const dim2 = gradient.dim2();
    tensor = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT,
                                tensorflow::TensorShape({dim1, dim2}));
    float* tensor_ptr = tensor.flat<float>().data();
    // uint8_t const* signs_ptr =
    //     reinterpret_cast<uint8_t const*>(gradient.signs().data());
    // int sign_begin = 0;
    PRINT_INFO;
    for (int i = 0; i < dim2; i++) {
        float const max = gradient.maxes(i);
        float const min = gradient.mins(i);
        uint8_t const* quantized_array = reinterpret_cast<uint8_t const*>(
            gradient.quantized_columns(i).data());
        float const positive_multiplier = (max) / (scope / 2);
        float const negative_multiplier = (-min) / (scope / 2);
        // std::cout << multiplier << std::endl;
        size_t begin = 0;
        for (size_t j = 0; j < dim1; j++) {
            uint32_t value = read_value(quantized_array, begin, level);
            // int const sign = read_value(signs_ptr, sign_begin, 1);
            float temp = -1;  //-1 does not mean anything
            if (value > (scope / 2)) {
                temp = (value - scope / 2) * positive_multiplier;
            } else {
                temp = - value * negative_multiplier;
            }
            // float temp = value * multiplier;
            // temp = (sign == 0) ? -temp : temp;
            tensor_ptr[dim2 * j + i] = temp;
            begin += level;
            // sign_begin++;
        }
    }
    PRINT_INFO;
}

void quantize_gradients_according_column(
    std::map<std::string, tensorflow::Tensor>& map_gradient,
    NamedGradientsAccordingColumn* named_gradients,
    int level,
    int threshold) {
    std::vector<std::pair<std::string, tensorflow::Tensor>> to_be_quantized;
    for (auto pair : map_gradient) {
        auto name = pair.first;
        auto& tensor = pair.second;
        auto size = tensor.NumElements();
        GradientAccordingColumn gac;
        if (size > threshold) {
            // gac.set_is_quantized(true);
            // quantize_gradient_according_column(level, tensor,
            // gac);
            to_be_quantized.push_back({name, tensor});
        } else {
            tensorflow::TensorProto tp;
            tensor.AsProtoField(&tp);
            gac.set_is_quantized(false);
            *gac.mutable_tensor() = tp;
            named_gradients->mutable_name_to_gradient()->insert({name, gac});
        }
    }
    int size = to_be_quantized.size();
    std::vector<GradientAccordingColumn> quantized_gradients;
    quantized_gradients.resize(size);
    std::vector<std::thread> threads;
    for (int i = 0; i < size; i++) {
        quantized_gradients[i].set_is_quantized(true);
        threads.push_back(std::thread(quantize_gradient_according_column, level,
                                      std::cref(to_be_quantized[i].second),
                                      std::ref(quantized_gradients[i])));
    }
    for (int i = 0; i < size; i++) {
        threads[i].join();
        named_gradients->mutable_name_to_gradient()->insert(
            {to_be_quantized[i].first, quantized_gradients[i]});
    }
}

void dequantize_gradients_according_column(
    NamedGradientsAccordingColumn& named_gradients,
    std::map<std::string, tensorflow::Tensor>& map_gradient) {
    auto& map = *named_gradients.mutable_name_to_gradient();
    std::vector<std::pair<std::string, GradientAccordingColumn>>
        to_be_dequantized;
    for (auto pair : map) {
        auto name = pair.first;
        auto& gradient = pair.second;
        bool is_quantized = gradient.is_quantized();
        tensorflow::Tensor tensor;
        if (is_quantized) {
            // dequantize_gradient_according_column(gradient,
            // tensor);
            to_be_dequantized.push_back({name, gradient});
        } else {
            tensor.FromProto(gradient.tensor());
            map_gradient.insert({name, tensor});
        }
    }
    int const size = to_be_dequantized.size();
    std::vector<tensorflow::Tensor> dequantized_gradients;
    dequantized_gradients.resize(size);
    std::vector<std::thread> threads;
    for (int i = 0; i < size; i++) {
        threads.push_back(std::thread(dequantize_gradient_according_column,
                                      std::cref(to_be_dequantized[i].second),
                                      std::ref(dequantized_gradients[i])));
    }
    for (int i = 0; i < size; i++) {
        threads[i].join();
        map_gradient.insert(
            {to_be_dequantized[i].first, dequantized_gradients[i]});
    }
}

}  // namespace split_by_0

}  // namespace adaptive_system
