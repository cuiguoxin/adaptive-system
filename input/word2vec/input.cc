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
#include "input/word2vec/input.h"

namespace adaptive_system {
	namespace {
		std::vector<std::pair<int, int>> training_data;
		std::unordered_map<std::string, int> word_to_index;
		
		bool less_compare_pair(std::pair<std::string, int> & a, std::pair<std::string, int> & b) {
			return a.second < b.second;
		}

		int get_index(std::string const & word) {
			auto iter = word_to_index.find(word);
			if (iter == word_to_index.end()) {
				return 0;
			}
			return iter->second;
		}

		void generate_according_one_line(std::vector<std::string> const & line_words) {
			const int window = 1;
			const size_t size = line_words.size();
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

		void generate_training_data(std::vector<std::vector<std::string>> const & raw_data) {
			size_t size = raw_data.size();
			for (int i = 0; i < size; i++) {
				std::vector<std::string> const & line_words = raw_data[i];
				generate_according_one_line(line_words);
			}
		}
	}
	void init() {
		std::string const raw_data_path = "";
		std::ifstream input_stream(raw_data_path);
		std::string line;
		std::unordered_map<std::string, int> word_count;
		std::vector<std::vector<std::string>> raw_data;
		while (std::getline(input_stream, line)) {
			std::vector<std::string> words_line;
			std::istringstream iss(line);
			std::string word;
			while (iss >> word) {
				if (word_count.find(word) == word_count.end()) {
					word_count[word] = 0;
				}
				word_count[word]++;
				words_line.push_back(word);
			}
			raw_data.push_back(words_line);
		}
		std::cout << "total distinct word size is " << word_count.size() << std::endl;
		size_t const k = 50000;
		std::vector<std::pair<std::string, int>> top_k;
		auto begin = word_count.begin();
		auto end = word_count.end();
		std::for_each(begin, end, [&top_k](std::pair<std::string, int> & pair) {
			top_k.push_back(pair);
		});
		//sort top_k
		std::sort(top_k.begin(), top_k.end(), less_compare_pair);
		size_t const size = top_k.size();
		size_t tail_sum = 0;
		for (int i = k; i < size; i++) {
			tail_sum += top_k[i].second;
		}
		word_to_index.insert(std::make_pair("UNK", 0));
		for (int i = 0; i < k; i++) {
			word_to_index.insert(std::make_pair(top_k[i].first, word_to_index.size()));
		}
		top_k.clear();
		generate_training_data(raw_data);
	}

	std::pair<tensorflow::Tensor, tensorflow::Tensor> get_next_batch() {
		static const batch_size = 256;
		static size_t current_index = 0; 
		static const size_t total_training_data = training_data.size();
		tensorflow::Tensor batch_tensor(tensorflow::DataType::DT_INT32, 
			tensorflow::TensorShape({ batch_size }));
		float* batch_tensor_ptr = batch_tensor.flat<float>().data();
		tensorflow::Tensor label_tensor(tensorflow::DataType::DT_INT32,
			tensorflow::TensorShape({ batch_size, 1 }));
		float* label_tensor_ptr = label_tensor.flat<float>().data();
		for (int i = 0; i < batch_size; i++) {
			current_index = current_index % total_training_data;
			std::pair<int, int> const & current_pair = training_data[current_index];
			current_index++;
			//assign value
			batch_tensor_ptr[i] = current_pair.first;
			label_tensor_ptr[i] = current_pair.second;
		}
		return std::make_pair(batch_tensor, label_tensor);
	}

}