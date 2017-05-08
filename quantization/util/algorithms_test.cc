#include "quantization/util/algorithms.h"
#include <iostream>
using adaptive_system::quantize;
using adaptive_system::dequantize;
using std::cout;
using std::endl;
void test(adaptive_system::QUANTIZATION_TYPE type, float* array_float, float max, float min, size_t const array_length){
	tensorflow::uint8* quantized_data_p;
	tensorflow::uint8** quantized_data = &quantized_data_p;
	size_t quantized_data_length = -1;
	quantize(type,
		array_float,
		max,
		min,
		array_length,
		quantized_data,
		quantized_data_length
		);
	dequantize(type,
		  quantized_data_p,
		  quantized_data_length,
		  array_length,
		  max,
		  min,
		  array_float
		  );
	std::for_each(array_float, array_float + array_length, [](float value){ cout << value << " "; });
	cout << endl;
	delete [] quantized_data_p; 
}
int main(){
	float array_float[] = {1.0, 4.5, 3.7, -2.5, 8.0, 6.6, -1.0, 5.2, 3.2, 2.8};
	test(adaptive_system::QUANTIZATION_TYPE::FOUR_BIT, array_float, 8.0, -2.5, 10);
	float array_float1[] = {0.2, 0.1, 0.3, 0.25, 0.22};
	test(adaptive_system::QUANTIZATION_TYPE::TWO_BIT, array_float1, 0.3, 0.1, 5);
	float array_float2[] = {0.01, 0, 0, 0.002, -0.001, 0.004};
	test(adaptive_system::QUANTIZATION_TYPE::FOUR_BIT, array_float2, 0.01, -0.001, 6);
	return 0;
}
