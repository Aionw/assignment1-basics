#include "tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "mmapped_file.h"
#include "ylt/easylog.hpp"

namespace Tokenizer {
namespace {

using PairType = std::pair<WordItem, WordItem>;

template <typename Fn>
void forEachPair(const WordItems& word, Fn&& fn) {
    const auto& items = word.items();
    if (items.size() < 2) {
        return;
    }
    for (size_t i = 0; i + 1 < items.size(); ++i) {
        fn(PairType{items[i], items[i + 1]});
    }
}

bool pairIsLarger(const PairType& l, const PairType& r) {
    if (l.first == r.first) {
        return l.second.value() > r.second.value();
    }
    return l.first.value() > r.first.value();
}

struct PairHeapEntry {
    PairType pair;
    size_t count;
};

struct PairHeapLess {
    bool operator()(const PairHeapEntry& l, const PairHeapEntry& r) const {
        if (l.count != r.count) {
            return l.count < r.count;
        }
        return pairIsLarger(r.pair, l.pair);
    }
};

}  // namespace

WordItems::WordItems(std::string_view word) {
    items_.reserve(word.size());
    for (int i = 0; i < word.size(); ++i) {
        items_.emplace_back(word.data(), i);
    }
}

std::vector<std::pair<WordItem, WordItem>> WordItems::pairs() const {
    if (items_.size() < 2) {
        return {};
    }
    std::vector<std::pair<WordItem, WordItem>> out{};
    out.reserve(items_.size() - 1);
    for (int i = 0; i < items_.size() - 1; ++i) {
        out.emplace_back(items_[i], items_[i + 1]);
    }
    return out;
}

void WordItems::merge(const WordItemPair& pair) {
    if (items_.empty()) {
        return;
    }
    std::vector<WordItem> items{};
    items.reserve(items_.size());
    size_t i = 0;
    while (i < items_.size() - 1) {
        if (items_[i] == pair.first && items_[i + 1] == pair.second) {
            items.emplace_back(items_[i] + items_[i + 1]);
            i += 2;
        } else {
            items.emplace_back(items_[i++]);
        }
    }
    if (i == items_.size() - 1) {
        items.emplace_back(items_[i]);
    }
    std::swap(items, items_);
}

BPETrainer::BPETrainer(size_t thread_count, const std::string& file_path,
                       const std::vector<std::string>& special_tokens, size_t vocab_size)
    : pool_(thread_count),
      file_(file_path, MmapAccess::read),
      vocab_size_(vocab_size),
      pre_tokenizer_(pool_, kGPT2Pattern.data(), special_tokens) {}

void BPETrainer::train() {
    auto start = std::chrono::high_resolution_clock::now();

    auto tokenize_start = std::chrono::high_resolution_clock::now();
    auto pre_tokens = pre_tokenizer_.tokenize(file_);
    auto tokenize_end = std::chrono::high_resolution_clock::now();
    if (!pre_tokens.has_value()) {
        return;
    }

    auto word_init_start = std::chrono::high_resolution_clock::now();
    absl::flat_hash_map<std::string_view, size_t> pretoken_counts = pre_tokens.value();
    std::vector<WordItems> word_items{};
    std::vector<size_t> word_counts{};
    word_items.reserve(pretoken_counts.size());
    word_counts.reserve(pretoken_counts.size());
    for (const auto& [word, count] : pretoken_counts) {
        word_items.emplace_back(word);
        word_counts.emplace_back(count);
    }
    auto word_init_end = std::chrono::high_resolution_clock::now();

    auto pair_init_start = std::chrono::high_resolution_clock::now();
    absl::flat_hash_map<PairType, size_t, WordItemPairHash> pair_count{};
    absl::flat_hash_map<PairType, absl::flat_hash_set<size_t>, WordItemPairHash>
        pair_to_word{};
    size_t initial_pair_capacity{0};
    for (const auto& items : word_items) {
        if (items.items().size() > 1) {
            initial_pair_capacity += items.items().size() - 1;
        }
    }
    pair_count.reserve(initial_pair_capacity);
    pair_to_word.reserve(initial_pair_capacity);
    for (size_t word_id = 0; word_id < word_items.size(); ++word_id) {
        const auto& items = word_items[word_id];
        forEachPair(items, [&](const PairType& p) {
            pair_count[p] += word_counts[word_id];
            pair_to_word[p].emplace(word_id);
        });
    }
    auto pair_init_end = std::chrono::high_resolution_clock::now();

    auto heap_init_start = std::chrono::high_resolution_clock::now();
    size_t vocab_capacity = vocab_size_ - pre_tokenizer_.getSpecialTokens().size() - 256;
    merges_.reserve(vocab_capacity);
    std::priority_queue<PairHeapEntry, std::vector<PairHeapEntry>, PairHeapLess> pair_heap{};
    for (const auto& [p, c] : pair_count) {
        pair_heap.push(PairHeapEntry{p, c});
    }
    auto heap_init_end = std::chrono::high_resolution_clock::now();

    auto pushPairSnapshot = [&pair_count, &pair_heap](const PairType& p) {
        auto it = pair_count.find(p);
        if (it != pair_count.end() && it->second > 0) {
            pair_heap.push(PairHeapEntry{p, it->second});
        }
    };

    long long heap_pop_us{0};
    long long update_us{0};
    size_t stale_heap_entries{0};
    size_t valid_heap_entries{0};
    size_t affected_word_visits{0};
    size_t touched_pair_visits{0};
    size_t max_affected_words{0};
    size_t max_touched_pairs{0};
    auto merge_loop_start = std::chrono::high_resolution_clock::now();
    while (vocab_capacity != 0) {
        if (pair_count.empty()) {
            break;
        }
        std::optional<PairType> max_pair{};
        auto heap_pop_start = std::chrono::high_resolution_clock::now();
        while (!pair_heap.empty()) {
            auto top = pair_heap.top();
            pair_heap.pop();
            auto current = pair_count.find(top.pair);
            if (current != pair_count.end() && current->second == top.count) {
                max_pair = std::move(top.pair);
                valid_heap_entries++;
                break;
            }
            stale_heap_entries++;
        }
        auto heap_pop_end = std::chrono::high_resolution_clock::now();
        heap_pop_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(heap_pop_end - heap_pop_start)
                .count();
        if (!max_pair) {
            break;
        }

        auto update_start = std::chrono::high_resolution_clock::now();
        merges_.emplace_back(*max_pair);
        auto max_words_it = pair_to_word.find(*max_pair);
        if (max_words_it == pair_to_word.end()) {
            break;
        }
        auto max_pair_words = std::move(max_words_it->second);
        pair_to_word.erase(max_words_it);
        absl::flat_hash_set<PairType, WordItemPairHash> touched_pairs{};
        affected_word_visits += max_pair_words.size();
        max_affected_words = std::max(max_affected_words, max_pair_words.size());
        for (const auto word_id : max_pair_words) {
            auto& items = word_items[word_id];
            size_t wc = word_counts[word_id];
            forEachPair(items, [&](const PairType& p) {
                auto count_it = pair_count.find(p);
                if (count_it == pair_count.end()) {
                    return;
                }
                count_it->second -= wc;
                auto words_it = pair_to_word.find(p);
                if (words_it != pair_to_word.end()) {
                    words_it->second.erase(word_id);
                }
                if (count_it->second == 0) {
                    pair_count.erase(count_it);
                    if (words_it != pair_to_word.end()) {
                        pair_to_word.erase(words_it);
                    }
                }
                touched_pairs.emplace(p);
            });
            items.mergeAndVisitNewPairs(*max_pair, [&](const PairType& p) {
                pair_count[p] += wc;
                pair_to_word[p].emplace(word_id);
                touched_pairs.emplace(p);
            });
        }
        for (const auto& p : touched_pairs) {
            pushPairSnapshot(p);
        }
        touched_pair_visits += touched_pairs.size();
        max_touched_pairs = std::max(max_touched_pairs, touched_pairs.size());
        auto update_end = std::chrono::high_resolution_clock::now();
        update_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(update_end - update_start).count();
        
        vocab_capacity--;
    }
    auto merge_loop_end = std::chrono::high_resolution_clock::now();
    ELOGFMT(WARN,
            "trainer profile: tokenize={}ms word_init={}ms pair_init={}ms heap_init={}ms "
            "merge_loop={}ms heap_pop={}ms update={}ms merges={} words={} pairs={} "
            "heap_valid={} heap_stale={} affected_words={} max_affected={} touched_pairs={} "
            "max_touched={} heap_size_end={}",
            std::chrono::duration_cast<std::chrono::milliseconds>(tokenize_end - tokenize_start)
                .count(),
            std::chrono::duration_cast<std::chrono::milliseconds>(word_init_end - word_init_start)
                .count(),
            std::chrono::duration_cast<std::chrono::milliseconds>(pair_init_end - pair_init_start)
                .count(),
            std::chrono::duration_cast<std::chrono::milliseconds>(heap_init_end - heap_init_start)
                .count(),
            std::chrono::duration_cast<std::chrono::milliseconds>(merge_loop_end - merge_loop_start)
                .count(),
            heap_pop_us / 1000, update_us / 1000, merges_.size(), word_items.size(),
            pair_count.size(), valid_heap_entries, stale_heap_entries, affected_word_visits,
            max_affected_words, touched_pair_visits, max_touched_pairs, pair_heap.size());
    ELOGFMT(WARN, "total train time: {}us",
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start)
                .count());
}

absl::flat_hash_map<size_t, std::string> BPETrainer::vocab() const {
    absl::flat_hash_map<size_t, std::string> r{};
    size_t offset{0};
    r.reserve(256 + pre_tokenizer_.getSpecialTokens().size() + merges_.size());
    for (const auto& t : pre_tokenizer_.getSpecialTokens()) {
        r[offset++] = std::string(t);
    }
    for (size_t i = 0; i < 256; ++i) {
        r[offset++] = std::string(1, static_cast<char>(i));
    }
    for (const auto& m : merges_) {
        r[offset++] = std::string((m.first + m.second).value());
    }
    return r;
}

std::vector<std::pair<std::string, std::string>> BPETrainer::merges() const {
    std::vector<std::pair<std::string, std::string>> r{};
    r.reserve(merges_.size());
    for (const auto& m : merges_) {
        r.emplace_back(std::string(m.first.value()), std::string(m.second.value()));
    }
    return r;
}

}  // namespace Tokenizer
