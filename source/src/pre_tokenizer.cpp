#include "pre_tokenizer.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "fmt/core.h"
#include "fmt/format.h"
#include "mmapped_file.h"
#include "pcre2_regex.h"
#include "thread_pool.h"
#include "ylt/easylog.hpp"
#include "ylt/util/expected.hpp"

namespace Tokenizer {

std::vector<std::vector<std::string_view>> distributeToThreads(
    const std::vector<std::string_view>& items, size_t thread_count) {
    std::vector<std::vector<std::string_view>> thread_items(thread_count);

    // Min-heap: (total_size, thread_index)
    std::vector<std::pair<size_t, size_t>> heap;
    heap.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        heap.emplace_back(0, i);
    }
    std::make_heap(heap.begin(), heap.end(), std::greater<>{});

    // Assign each item to thread with smallest load
    for (const auto& item : items) {
        std::pop_heap(heap.begin(), heap.end(), std::greater<>{});
        auto& [min_size, min_thread] = heap.back();
        thread_items[min_thread].push_back(item);
        min_size += item.size();
        std::push_heap(heap.begin(), heap.end(), std::greater<>{});
    }

    return thread_items;
}

ylt::expected<absl::flat_hash_map<std::string_view, size_t>, std::string> PreTokenizer::tokenize(
    MmappedFile& file) {
    if (!re_) {
        return ylt::unexpected<std::string>(re_.error());
    }

    // Split file by special tokens
    std::vector<std::string_view> input_items{file.data()};
    std::vector<std::string_view> output_items{};
    if (!special_tokens_.empty()) {
        for (auto sep : special_tokens_) {
            for (auto item : input_items) {
                std::vector<std::string_view> splitted = absl::StrSplit(item, sep);
                output_items.insert(output_items.end(), splitted.begin(), splitted.end());
            }
            input_items.swap(output_items);
            output_items.clear();
        }
    }

    // Distribute tokens to threads using min-heap based on total token size
    std::vector<std::vector<std::string_view>> thread_items =
        distributeToThreads(input_items, pool_.threadCount());

    // Submit tasks to thread pool
    absl::flat_hash_map<std::string_view, size_t> pre_tokens{};
    std::mutex pre_token_mu{};
    auto start = std::chrono::steady_clock::now();

    for (size_t t = 0; t < pool_.threadCount(); ++t) {
        pool_.submit([this, &pre_token_mu, &pre_tokens, &thread_items, t] {
            auto find_start = std::chrono::steady_clock::now();
            absl::flat_hash_map<std::string_view, size_t> token_count{};
            for (const auto& item : thread_items[t]) {
                re_.findAll(item, [&token_count](std::string_view match) { token_count[match]++; });
            }
            auto find_end = std::chrono::steady_clock::now();
            auto find_us =
                std::chrono::duration_cast<std::chrono::microseconds>(find_end - find_start);

            std::lock_guard lk(pre_token_mu);
            auto assign_start = std::chrono::steady_clock::now();
            for (auto [k, v] : token_count) {
                pre_tokens[k] += v;
            }
            auto assign_end = std::chrono::steady_clock::now();
            auto assign_us =
                std::chrono::duration_cast<std::chrono::microseconds>(assign_end - assign_start);
        });
    }

    pool_.wait();
    auto end = std::chrono::steady_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    ELOGFMT(WARN, "total tokenization: {}ms", total_us.count());
    return {std::move(pre_tokens)};
}

}  // namespace Tokenizer
