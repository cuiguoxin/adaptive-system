#include "quantization/util/algorithms.h"

namespace adaptive_system {

namespace {
const float eps = 0.000001f;
void quantize_less_8_bits(
    const QUANTIZATION_TYPE type, float* raw_data, const float max_value,
    const float min_value, const size_t raw_data_length,
    tensorflow::uint8** quantized_data,
    size_t& quantized_data_length  // the number of bytes to output
    ) {
  const int q_type = static_cast<int>(type);  // for example 2
  const int scope = std::pow(2, q_type);
  const float multiplizer = scope / (max_value + eps - min_value);
  std::for_each(raw_data, raw_data + raw_data_length,
                [multiplizer, min_value](float& ref) {
                  ref = (ref - min_value) * multiplizer;
                });
  const int length_per_iter = 8 / q_type;  // for example 4
  quantized_data_length = static_cast<size_t>(
      std::ceil(raw_data_length / static_cast<float>(length_per_iter)));
  std::cout << "quantized_data_length is " << quantized_data_length
            << std::endl;
  tensorflow::uint8* output = new tensorflow::uint8[quantized_data_length];
  for (int i = 0; i < quantized_data_length; i++) {
    output[i] = 0;
    for (int j = 0; j < length_per_iter; j++) {
      const int index_for_raw = length_per_iter * i + j;
      if (index_for_raw >= raw_data_length) {
        break;
      }
      tensorflow::uint8 value_raw = raw_data[index_for_raw];
      output[i] = output[i] | (value_raw << (q_type * j));
    }
  }
  *quantized_data = output;
}
// tensor has been allocated
void quantize_greater_8_bits(const QUANTIZATION_TYPE type,
                             tensorflow::Tensor& raw_data,
                             const float max_value, const float min_value,
                             tensorflow::Tensor& tensor) {
  float scale_factor = 0.0;
  if (type == QUANTIZATION_TYPE::EIGHT_BIT) {
    scale_factor =
        (static_cast<double>(std::numeric_limits<tensorflow::uint8>::max()) -
         static_cast<double>(std::numeric_limits<tensorflow::uint8>::min())) /
        (max_value - min_value);
    auto o = tensor.flat<tensorflow::uint8>();
    o = ((raw_data.flat<float>().cwiseMin(max_value).cwiseMax(min_value) -
          min_value) *
             scale_factor +
         0.5f)
            .cast<tensorflow::uint8>();
  } else if (type == QUANTIZATION_TYPE::SIXTEEN_BIT) {
    scale_factor =
        (static_cast<double>(std::numeric_limits<tensorflow::uint16>::max()) -
         static_cast<double>(std::numeric_limits<tensorflow::uint16>::min())) /
        (max_value - min_value);
    auto o = tensor.flat<tensorflow::uint16>();
    o = ((raw_data.flat<float>().cwiseMin(max_value).cwiseMax(min_value) -
          min_value) *
             scale_factor +
         0.5f)
            .cast<tensorflow::uint16>();
  }
}

void dequantize_greater_8_bits(const QUANTIZATION_TYPE type,
                               tensorflow::Tensor& quantized_data,
                               const float max_range, const float min_range,
                               tensorflow::Tensor& tensor) {
  if (type == QUANTIZATION_TYPE::EIGHT_BIT) {
    const float scale_factor =
        (max_range - min_range) /
        (static_cast<float>(std::numeric_limits<tensorflow::uint8>::max()) -
         std::numeric_limits<tensorflow::uint8>::min());

    float* out_ptr = tensor.flat<float>().data();
    const tensorflow::uint8* in_ptr =
        quantized_data.flat<tensorflow::uint8>().data();

    const int num_elements = quantized_data.NumElements();
    for (int i = 0; i < num_elements; ++i) {
      out_ptr[i] = ((static_cast<int>(in_ptr[i])) * scale_factor) + min_range;
    }
  } else if (type == QUANTIZATION_TYPE::SIXTEEN_BIT) {
    const float scale_factor =
        (max_range - min_range) /
        (static_cast<float>(std::numeric_limits<tensorflow::uint16>::max()) -
         std::numeric_limits<tensorflow::uint16>::min());

    float* out_ptr = tensor->flat<float>().data();
    const tensorflow::uint16* in_ptr =
        quantized_data.flat<tensorflow::uint16>().data();

    const int num_elements = input.NumElements();
    for (int i = 0; i < num_elements; ++i) {
      out_ptr[i] = ((static_cast<int>(in_ptr[i])) * scale_factor) + min_range;
    }
  }
}
void dequantize_less_8_bits(const QUANTIZATION_TYPE type,
                            const tensorflow::uint8* quantized_data,
                            const size_t raw_data_length, const float max_value,
                            const float min_value, float* raw_data) {
  static const tensorflow::uint8 mask_2_bits = 3, mask_4_bits = 15,
                                 mask_8_bits = 255;
  const int q_type = static_cast<int>(type);  // for example 2
  std::cout << "q type is " << q_type << std::endl;
  const int scope = std::pow(2, q_type);
  const int length_per_iter = 8 / q_type;  // for example 4
  const float multiplier = (max_value - min_value) / scope;
  std::cout << "multiplier is " << multiplier << std::endl;
  int i = 0;
  std::function<void(float&)> func = [=, &i](float& ref) {
    const int index_for_q_data = i / length_per_iter;
    const int index_in_iter = i - index_for_q_data * length_per_iter;
    tensorflow::uint8 q_data = quantized_data[index_for_q_data];
    const int move_right = q_type * index_in_iter;
    q_data = q_data >> move_right;
    switch (q_type) {
      case 2:
        q_data &= mask_2_bits;
        break;
      case 4:
        q_data &= mask_4_bits;
        break;
      case 8:
        q_data &= mask_8_bits;
        break;
    }
    ref = q_data * multiplier + min_value + multiplier * 0.5;
    ++i;
  };
  std::for_each(raw_data, raw_data + raw_data_length, func);
}
GRAD_QUANT_LEVEL cast_quantization_type_to_grad_quant_level(
    QUANTIZATION_TYPE type) {
  switch (type) {
    case QUANTIZATION_TYPE::TWO_BIT:
      return GRAD_QUANT_LEVEL::TWO;
    case QUANTIZATION_TYPE::FOUR_BIT:
      return GRAD_QUANT_LEVEL::FOUR;
    case QUANTIZATION_TYPE::EIGHT_BIT:
      return GRAD_QUANT_LEVEL::EIGHT;
    case QUANTIZATION_TYPE::SIXTEEN_BIT:
      return GRAD_QUANT_LEVEL::SIXTEEN;
  }
  return GRAD_QUANT_LEVEL::NONE;
}
tensorflow::DataType cast_quantization_type_to_data_type(
    QUANTIZATION_TYPE type) {
  if (type == QUANTIZATION_TYPE::EIGHT_BIT) {
    return tensorflow::DataType::DT_UINT8;
  } else if (type == QUANTIZATION_TYPE::SIXTEEN_BIT) {
    return tensorflow::DataType::DT_UINT16;
  }
  std::terminate();
  return tensorflow::DataType::DT_INVALID;
}
}
// raw_tensor ---->>> grad
void quantize(const QUANTIZATION_TYPE type,
              tensorflow::Tensor const& raw_tensor, float const max_value,
              float const min_value, Gradient& grad) {
  grad.set_max(max_value);
  grad.set_min(min_value);
  grad.set_level = cast_quantization_type_to_grad_quant_level(type);
  if (type == QUANTIZATION_TYPE::TWO_BIT ||
      type == QUANTIZATION_TYPE::FOUR_BIT) {
    float* raw_ptr = raw_tensor.flat<float>().data();
    int num_elements = raw_tensor.NumElements();
    tensorflow::uint8* out_ptr;
    size_t out_ptr_length = 0;
    quantize_less_8_bits(type, raw_ptr, max_value, min_value, num_elements,
                         &out_ptr, out_ptr_length);
    grad.set_tensor_le_8(out_ptr, out_ptr_length);
    delete[] out_ptr;
  } else if (type == QUANTIZATION_TYPE::EIGHT_BIT ||
             type == QUANTIZATION_TYPE::SIXTEEN_BIT) {
    tensorflow::Tensor tensor(cast_quantization_type_to_data_type(type),
                              raw_data.shape());
    quantize_greater_8_bits(type, raw_tensor, max_value, min_value, tensor);
    tensorflow::TensorProto* tensor_proto = new tensorflow::TensorProto;
    tensor.AsProtoField(tensor_proto);
    grad.set_allocated_tensor_ge_8(tensor_proto);
  }
}

// grad --->>> raw_tensor
// raw_tensor can be an empty tensor
void dequantize(const QUANTIZATION_TYPE type, Gradient& grad,
                tensorflow::Tensor& raw_tensor) {
  float const max_value = grad.max();
  float const min_value = grad.min();
  if (type == QUANTIZATION_TYPE::TWO_BIT ||
      type == QUANTIZATION_TYPE::FOUR_BIT) {
    tensorflow::TensorShapeProto* tensor_shape_proto =
        grad.mutable_tensor_shape();
    tensorflow::TensorShape tensor_shape(tensor_shape_proto);
    raw_tensor =
        tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, tensor_shape);
    float* raw_ptr = raw_tensor.flat<float>().data();
    int number_raw = raw_tensor.NumElements();

    tensorflow::uint8 const* quantized_ptr = grad.mutable_le_8()->data();
    dequantize_less_8_bits(type, quantized_ptr, number_raw, max_value,
                           min_value, raw_ptr);
  } else if (type == QUANTIZATION_TYPE::EIGHT_BIT ||
             type == QUANTIZATION_TYPE::SIXTEEN_BIT) {
    tensorflow::TensorProto& tensor_quantized_proto =
        *grad.mutable_tensor_ge_8();
    tensorflow::Tensor temp;
    temp.FromProto(tensor_quantized_proto);
    // must allocate tensor memory outside dequantize_greater_8_bits
    raw_tensor = tensorflow::Tensor(DataType::DT_FLOAT, temp.shape());
    dequantize_greater_8_bits(type, temp, max_value, min_value, raw_tensor);
  }
}
}
