#include <sstream>
#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>

#include "input/word2vec/input.h"

namespace adaptive_system {

	namespace word2vec {

		namespace {
			std::deque<std::pair<int32_t, int32_t>> training_data;
			google::protobuf::Map<std::string, int32_t>  word_to_index;
			std::ifstream stream;

			int get_index(std::string const & word) {
				auto iter = word_to_index.find(word);
				if (iter == word_to_index.end()) {
					return 0;
				}
				return iter->second;
			}

			void generate_according_one_line(std::vector<std::string> const & line_words) {
				const int window = 1;
				const int size = line_words.size();
				for (int i = window; i < size - window; i++) {
					size_t index_of_i = get_index(line_words[i]);
					//previous
					for (int j = i - window; j < i; j++) {
						size_t index_of_j = get_index(line_words[j]);
						training_data.push_back(std::make_pair(index_of_i, index_of_j));
					}
					//behind
					for (int j = i + 1; j <= i + window; j++) {
						size_t index_of_j = get_index(line_words[j]);
						training_data.push_back(std::make_pair(index_of_i, index_of_j));
					}
				}
			}		
		}
		void init(std::string const & raw_data_path,
			google::protobuf::Map<std::string, int32_t> const & word_2_index) {
			word_to_index = word_2_index;
			//std::ifstream input_stream(raw_data_path);
			stream.open(raw_data_path);
		}

		std::pair<tensorflow::Tensor, tensorflow::Tensor> get_next_batch(size_t const batch_size) {
			while (training_data.size() < batch_size) {
				std::string line;
				while (!std::getline(stream, line)) {
					strean.clear();
					stream.seekg(0, stream.beg);
				}
				std::istringstream iss(line);
				std::string word;
				std::vector<std::string> line_words;
				while (iss >> word) {
					line_words.push_back(word);
				}
				generate_according_one_line(line_words);
			}
			tensorflow::Tensor batch_tensor(tensorflow::DataType::DT_INT32,
				tensorflow::TensorShape({ (int)batch_size }));
			tensorflow::int32* batch_tensor_ptr = batch_tensor.flat<tensorflow::int32>().data();
			tensorflow::Tensor label_tensor(tensorflow::DataType::DT_INT32,
				tensorflow::TensorShape({ (int)batch_size, 1 }));
			tensorflow::int32* label_tensor_ptr = label_tensor.flat<tensorflow::int32>().data();
			for (int i = 0; i < batch_size; i++) {
				std::pair<int, int> const current_pair = training_data.front();
				training_data.pop_front();
				//assign value
				batch_tensor_ptr[i] = current_pair.first;
				label_tensor_ptr[i] = current_pair.second;
			}
			return std::make_pair(batch_tensor, label_tensor);
		}

	}


}
