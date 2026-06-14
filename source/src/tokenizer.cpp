#include "tokenizer.h"

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

    auto pre_tokens = pre_tokenizer_.tokenize(file_);
    if (!pre_tokens.has_value()) {
        return;
    }

    absl::flat_hash_map<std::string_view, size_t> word_count = pre_tokens.value();
    absl::flat_hash_map<std::string_view, WordItems> word_items{};
    word_items.reserve(word_count.size());
    for (const auto& [k, _] : word_count) {
        word_items.emplace(k, k);
    }
    absl::flat_hash_map<PairType, size_t, WordItemPairHash> pair_count{};
    absl::flat_hash_map<PairType, absl::flat_hash_set<std::string_view>, WordItemPairHash>
        pair_to_word{};
    size_t initial_pair_capacity{0};
    for (const auto& [_, items] : word_items) {
        if (items.items().size() > 1) {
            initial_pair_capacity += items.items().size() - 1;
        }
    }
    pair_count.reserve(initial_pair_capacity);
    pair_to_word.reserve(initial_pair_capacity);
    for (const auto& [k, items] : word_items) {
        forEachPair(items, [&](const PairType& p) {
            pair_count[p] += word_count[k];
            pair_to_word[p].emplace(k);
        });
    }

    size_t vocab_capacity = vocab_size_ - pre_tokenizer_.getSpecialTokens().size() - 256;
    merges_.reserve(vocab_capacity);
    std::priority_queue<PairHeapEntry, std::vector<PairHeapEntry>, PairHeapLess> pair_heap{};
    for (const auto& [p, c] : pair_count) {
        pair_heap.push(PairHeapEntry{p, c});
    }

    auto pushPairSnapshot = [&pair_count, &pair_heap](const PairType& p) {
        auto it = pair_count.find(p);
        if (it != pair_count.end() && it->second > 0) {
            pair_heap.push(PairHeapEntry{p, it->second});
        }
    };

    while (vocab_capacity != 0) {
        if (pair_count.empty()) {
            break;
        }
        std::optional<PairType> max_pair{};
        while (!pair_heap.empty()) {
            auto top = pair_heap.top();
            pair_heap.pop();
            auto current = pair_count.find(top.pair);
            if (current != pair_count.end() && current->second == top.count) {
                max_pair = std::move(top.pair);
                break;
            }
        }
        if (!max_pair) {
            break;
        }

        merges_.emplace_back(*max_pair);
        auto max_words_it = pair_to_word.find(*max_pair);
        if (max_words_it == pair_to_word.end()) {
            break;
        }
        auto max_pair_words = std::move(max_words_it->second);
        pair_to_word.erase(max_words_it);
        absl::flat_hash_set<PairType, WordItemPairHash> touched_pairs{};
        for (const auto& w : max_pair_words) {
            auto& items = word_items.at(w);
            size_t wc = word_count[w];
            forEachPair(items, [&](const PairType& p) {
                auto count_it = pair_count.find(p);
                if (count_it == pair_count.end()) {
                    return;
                }
                count_it->second -= wc;
                auto words_it = pair_to_word.find(p);
                if (words_it != pair_to_word.end()) {
                    words_it->second.erase(w);
                }
                if (count_it->second == 0) {
                    pair_count.erase(count_it);
                    if (words_it != pair_to_word.end()) {
                        pair_to_word.erase(words_it);
                    }
                }
                touched_pairs.emplace(p);
            });
            items.merge(*max_pair);
            forEachPair(items, [&](const PairType& p) {
                pair_count[p] += wc;
                pair_to_word[p].emplace(w);
                touched_pairs.emplace(p);
            });
        }
        for (const auto& p : touched_pairs) {
            pushPairSnapshot(p);
        }
        
        vocab_capacity--;
    }
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
