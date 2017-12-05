#include "quantization/util/any_level.h"
#include <limits>
#include <thread>
#include <utility>
#include "quantization/util/helper.h"

namespace adaptive_system {
namespace {

void get_max_and_min_value(tensorflow::Tensor const& tensor,
                           float& max,
                           float& min) {
    float const* tensor_ptr = tensor.flat<float>().data();
    size_t size = tensor.NumElements();
    max = tensor_ptr[0];
    min = tensor_ptr[0];
    std::for_each(tensor_ptr, tensor_ptr + size,
                  [&max, &min](float const current) {
                      if (max < current) {
                          max = current;
                          return;
                      }
                      if (min > current) {
                          min = current;
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

void quantize_gradient(uint32_t const level,
                       tensorflow::Tensor const& tensor,
                       Gradient& gradient) {
    // gradient.Clear();
    size_t size = tensor.NumElements();  // number of float to be quantized
    float const* tensor_ptr = tensor.flat<float>().data();
    float max = 0, min = 0;
    get_max_and_min_value(tensor, max, min);
    gradient.set_quantization_level(level);
    gradient.set_max(max);
    gradient.set_min(min);
    tensor.shape().AsProto(gradient.mutable_tensor_shape());

    size_t quantized_size =
        std::ceil(((float)size) * level / 8);  // number of byte
    uint8_t* quantized_data = new uint8_t[quantized_size]();
    unsigned long long const scope = ((long long)1) << level;
    float const eps = 0.000001;
    float const multiplier = scope / (max + eps - min);
    size_t begin = 0;
    for (size_t i = 0; i < size; i++) {
        uint64_t value = multiplier * (tensor_ptr[i] - min);
        if (value == scope) {
            value--;
        }
        set_value(quantized_data, begin, level, value);
        begin += level;
    }
    gradient.set_quantized_tensor(quantized_data, quantized_size);
    delete[] quantized_data;
}

void dequantize_gradient(Gradient const& gradient, tensorflow::Tensor& tensor) {
    tensorflow::TensorShape tensor_shape(gradient.tensor_shape());
    tensor = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, tensor_shape);
    float* tensor_ptr = tensor.flat<float>().data();
    uint8_t const* quantized_array =
        reinterpret_cast<uint8_t const*>(gradient.quantized_tensor().data());
    size_t size = tensor_shape.num_elements();
    uint32_t const level = gradient.quantization_level();
    unsigned long long const scope = ((long long)1) << level;
    float const max = gradient.max(), min = gradient.min();
    float const multiplier = (max - min) / scope;
    size_t begin = 0;
    for (size_t i = 0; i < size; i++) {
        uint32_t value = read_value(quantized_array, begin, level);
        tensor_ptr[i] = multiplier * value + min + multiplier / 2;
        begin += level;
    }
}

void quantize_gradients(std::map<std::string, tensorflow::Tensor>& map_gradient,
                        NamedGradients* named_gradients,
                        int level) {
    int const size = map_gradient.size();
    std::vector<Gradient> grads;
    std::vector<std::thread> threads;
    grads.resize(size);
    std::vector<std::string> variable_names;
    int index = 0;
    for (auto iter = map_gradient.begin(); iter != map_gradient.end(); iter++) {
        std::string const& variable_name = iter->first;
        variable_names.push_back(variable_name);
        tensorflow::Tensor& raw_tensor = iter->second;
        threads.push_back(std::thread(quantize_gradient, level,
                                      std::ref(raw_tensor),
                                      std::ref(grads[index++])));
    }
    for (int i = 0; i < size; i++) {
        threads[i].join();
        named_gradients->mutable_name_to_gradient()->insert(
            google::protobuf::MapPair<std::string, Gradient>(variable_names[i],
                                                             grads[i]));
    }
}

void dequantize_gradients(
    NamedGradients& named_gradients,
    std::map<std::string, tensorflow::Tensor>& map_gradient) {
    int const size = named_gradients.name_to_gradient().size();
    std::vector<tensorflow::Tensor> tensors;
    std::vector<std::thread> threads;
    tensors.resize(size);
    std::vector<std::string> variable_names;
    int index = 0;
    for (auto iter = named_gradients.mutable_name_to_gradient()->begin();
         iter != named_gradients.mutable_name_to_gradient()->end(); iter++) {
        std::string const& variable_name = iter->first;
        variable_names.push_back(variable_name);
        Gradient& gradient = iter->second;
        threads.push_back(std::thread(dequantize_gradient, std::ref(gradient),
                                      std::ref(tensors[index++])));
    }
    for (int i = 0; i < size; i++) {
        threads[i].join();
        map_gradient.insert(std::pair<std::string, tensorflow::Tensor>(
            variable_names[i], tensors[i]));
    }
}

void quantize_gradient_according_column(uint32_t const level,
                                        tensorflow::Tensor const& tensor,
                                        GradientAccordingColumn& gradient) {
    auto dims = tensor.dims();
    if (dims != 2) {
        PRINT_ERROR_MESSAGE("dims is not 2");
        std::terminate();
    }
    int dim1 = tensor.dim_size(0);
    int dim2 = tensor.dim_size(1);
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
    unsigned long long const scope = ((long long)1) << level;
    float const eps = 0.000001;
    // quantize each column
    for (int i = 0; i < dim2; i++) {
        float* col_ptr = new float[dim1]();
        for (int j = 0; j < dim1; j++) {
            col_ptr[j] = tensor_ptr[dim2 * j + i];
        }
        size_t quantized_size =
            std::ceil(((float)dim1) * level / 8);  // number of byte
        uint8_t* quantized_data = new uint8_t[quantized_size]();
        float const max = max_vector[i], min = min_vector[i];
        float const multiplier = scope / (max + eps - min);
        size_t begin = 0;
        for (size_t j = 0; j < dim1; j++) {
            uint64_t value = multiplier * (col_ptr[j] - min);
            if (value == scope) {
                value--;
            }
            set_value(quantized_data, begin, level, value);
            begin += level;
        }
        gradient.add_maxes(max);
        gradient.add_mins(min);
        gradient.add_quantized_columns(quantized_data, quantized_size);
        delete[] quantized_data;
        delete[] col_ptr;
    }
    // assign to gradient
    gradient.set_dim1(dim1);
    gradient.set_dim2(dim2);
    gradient.set_quantization_level(level);
}

void dequantize_gradient_according_column(
    GradientAccordingColumn const& gradient,
    tensorflow::Tensor& tensor) {
    int const level = gradient.quantization_level();
    unsigned long long const scope = ((long long)1) << level;
    int const dim1 = gradient.dim1();
    int const dim2 = gradient.dim2();
    tensor = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT,
                                tensorflow::TensorShape({dim1, dim2}));
    float* tensor_ptr = tensor.flat<float>().data();

    for (int i = 0; i < dim2; i++) {
        float const max = gradient.maxes(i), min = gradient.mins(i);
        uint8_t const* quantized_array = reinterpret_cast<uint8_t const*>(
            gradient.quantized_columns(i).data());
        float const multiplier = (max - min) / scope;
        size_t begin = 0;
        for (size_t j = 0; j < dim1; j++) {
            uint32_t value = read_value(quantized_array, begin, level);
            tensor_ptr[dim2 * j + i] =
                multiplier * value + min + multiplier / 2;
            begin += level;
        }
    }
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
    int size = to_be_dequantized.size();
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
}  // namespace adaptive_system
