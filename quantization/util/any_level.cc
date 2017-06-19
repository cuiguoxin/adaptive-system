#include "quantization/util/any_level.h"

namespace adaptive_system {
	namespace {
		//0 <= start <= end <= 7, and assume that value is less than 2^(end - start)
		void set_byte(uint8_t * byte, size_t start, size_t end, uint8_t value) {
			uint8_t right_offset = 7 - end;
			uint8_t value_right_move = value << right_offset;
			*byte |= value_right_move;
		}

		void set_value(uint8_t * arr, size_t start, size_t length, uint32_t value) {
			size_t index_in_array_begin = start / 8;
			size_t index_in_byte_begin = start - index_in_array_begin * 8;
			size_t end = start + length - 1; // must minus 1
			size_t index_in_array_end = end / 8;
			size_t index_in_byte_end = end - index_in_array_end * 8;
			if (index_in_array_begin == index_in_array_end) {
				set_byte(arr + index_in_array_begin, index_in_byte_begin, index_in_byte_end, value);
			}
			else {
				size_t size = index_in_array_end - index_in_array_begin + 1;
				uint8_t* arr_size = new uint8_t[size]();
				std::fill(arr_size, arr_size + size, 8);
				arr_size[0] = 7 - index_in_byte_begin;
				arr_size[size - 1] = index_in_byte_end;
				for (size_t i = index_in_array_begin + 1; i < index_in_array_end; i++) {

				}
			}

		}
	}
}