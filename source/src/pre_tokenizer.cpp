#include "pre_tokenizer.h"

#include <algorithm>
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

    auto total_start = std::chrono::steady_clock::now();
    auto split_start = std::chrono::steady_clock::now();

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
    auto split_end = std::chrono::steady_clock::now();

    // Distribute tokens to threads using min-heap based on total token size
    auto distribute_start = std::chrono::steady_clock::now();
    std::vector<std::vector<std::string_view>> thread_items =
        distributeToThreads(input_items, pool_.threadCount());
    auto distribute_end = std::chrono::steady_clock::now();

    // Submit tasks to thread pool
    absl::flat_hash_map<std::string_view, size_t> pre_tokens{};
    std::mutex pre_token_mu{};
    std::vector<size_t> local_unique_counts(pool_.threadCount(), 0);
    std::vector<long long> find_times_us(pool_.threadCount(), 0);
    std::vector<long long> assign_times_us(pool_.threadCount(), 0);
    auto tokenize_start = std::chrono::steady_clock::now();

    for (size_t t = 0; t < pool_.threadCount(); ++t) {
        pool_.submit([this, &pre_token_mu, &pre_tokens, &thread_items, &local_unique_counts,
                      &find_times_us, &assign_times_us, t] {
            auto find_start = std::chrono::steady_clock::now();
            absl::flat_hash_map<std::string_view, size_t> token_count{};
            pcre2_match_data* match_data = re_.createMatchData();
            if (match_data != nullptr) {
                for (const auto& item : thread_items[t]) {
                    re_.findAll(
                        item, match_data,
                        [&token_count](std::string_view match) { token_count[match]++; });
                }
                Pcre2Regex::freeMatchData(match_data);
            }
            auto find_end = std::chrono::steady_clock::now();
            find_times_us[t] =
                std::chrono::duration_cast<std::chrono::microseconds>(find_end - find_start)
                    .count();
            local_unique_counts[t] = token_count.size();

            std::lock_guard lk(pre_token_mu);
            auto assign_start = std::chrono::steady_clock::now();
            for (auto [k, v] : token_count) {
                pre_tokens[k] += v;
            }
            auto assign_end = std::chrono::steady_clock::now();
            assign_times_us[t] =
                std::chrono::duration_cast<std::chrono::microseconds>(assign_end - assign_start)
                    .count();
        });
    }

    pool_.wait();
    auto tokenize_end = std::chrono::steady_clock::now();
    long long max_find_us{0};
    long long sum_find_us{0};
    long long max_assign_us{0};
    long long sum_assign_us{0};
    size_t summed_local_unique{0};
    for (size_t i = 0; i < pool_.threadCount(); ++i) {
        max_find_us = std::max(max_find_us, find_times_us[i]);
        sum_find_us += find_times_us[i];
        max_assign_us = std::max(max_assign_us, assign_times_us[i]);
        sum_assign_us += assign_times_us[i];
        summed_local_unique += local_unique_counts[i];
    }
    auto total_end = std::chrono::steady_clock::now();
    ELOGFMT(WARN,
            "pretokenize profile: split={}ms distribute={}ms regex_wall={}ms total={}ms "
            "chunks={} unique={} local_unique_sum={} find_sum={}ms find_max={}ms "
            "assign_sum={}ms assign_max={}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(split_end - split_start).count(),
            std::chrono::duration_cast<std::chrono::milliseconds>(distribute_end - distribute_start)
                .count(),
            std::chrono::duration_cast<std::chrono::milliseconds>(tokenize_end - tokenize_start)
                .count(),
            std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count(),
            input_items.size(), pre_tokens.size(), summed_local_unique, sum_find_us / 1000,
            max_find_us / 1000, sum_assign_us / 1000, max_assign_us / 1000);
    return {std::move(pre_tokens)};
}

}  // namespace Tokenizer
