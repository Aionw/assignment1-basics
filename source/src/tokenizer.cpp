#include "tokenizer.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/cleanup/cleanup.h"
#include "mmapped_file.h"
#include "ylt/easylog.hpp"

namespace Tokenizer {

WordItems::WordItems(std::string_view word) {
    for (int i = 0; i < word.size(); ++i) {
        items_.emplace_back(word.data(), i);
    }
}

std::vector<std::pair<WordItem, WordItem>> WordItems::pairs() const {
    if (items_.size() < 2) {
        return {};
    }
    std::vector<std::pair<WordItem, WordItem>> out{};
    for (int i = 0; i < items_.size() - 1; ++i) {
        out.emplace_back(items_[i], items_[i + 1]);
    }
    return out;
}

void WordItems::merge(const WordItemPair& pair) {
    if(items_.empty()) {return;}
    std::vector<WordItem> items{};
    size_t i = 0;
    while(i<items_.size()-1) {
        if(items_[i] == pair.first && items_[i+1] == pair.second) {
            items.emplace_back(items_[i] + items_[i+1]);
            i+=2;
        } else {
            items.emplace_back(items_[i++]);
        }
    }
    if(i == items_.size() - 1) {
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
    using PairType = std::pair<WordItem, WordItem>;
    auto start =  std::chrono::high_resolution_clock::now();

    auto pre_tokens = pre_tokenizer_.tokenize(file_);
    if (!pre_tokens.has_value()) {
        return;
    }

    absl::flat_hash_map<std::string_view, size_t> word_count = pre_tokens.value();
    absl::flat_hash_map<std::string_view, WordItems> word_items{};
    for (const auto& [k, _] : word_count) {
        word_items.emplace(k, k);
    }
    absl::flat_hash_map<PairType, size_t, WordItemPairHash> pair_count{};
    absl::flat_hash_map<PairType, absl::flat_hash_set<std::string_view>, WordItemPairHash> pair_to_word{};
    for (const auto& [k, items] : word_items) {
        for (const auto& p : items.pairs()) {
            pair_count[p] += word_count[k];
            pair_to_word[p].emplace(k);
       }
    }

    size_t vocab_capacity = vocab_size_ - pre_tokenizer_.getSpecialTokens().size() - 256;
    size_t total_merges = vocab_capacity;
    size_t current_merge = 0;
    size_t round{0};
    while (vocab_capacity != 0) {

        std::optional<PairType> max_pair{};
        size_t max_count{0};

        for (const auto& [k, items] : word_items) {
            for (const auto& p : items.pairs()) {
                if(pair_count[p] > max_count) {
                    max_count = pair_count[p];
                    max_pair = p;
                } else if(pair_count[p] == max_count) {
                    if(p.first == max_pair->first && p.second.value() > max_pair->second.value()) {
                        max_pair = p;
                    } else if(p.first != max_pair->first && p.first.value() > max_pair->first.value()) {
                        max_pair = p;
                    }
                }
            }
        }

        if (pair_count.empty()) {
            break;
        }

        merges_.emplace_back(*max_pair);
        const auto& max_pair_words = pair_to_word[*max_pair];
        for(const auto& w : max_pair_words) {
            auto& items = word_items.at(w);
            size_t wc = word_count[w];
            for(const auto& p :items.pairs()) {
                pair_count[p] -= wc;
                pair_to_word[p].erase(w);
                if(pair_count[p] == 0) {
                    pair_count.erase(p);
                }
            }
            items.merge(*max_pair);
            for(const auto& p :items.pairs()) {
                pair_count[p] += wc;
                pair_to_word[p].emplace(w);
            }
        }
        auto merge_end = std::chrono::high_resolution_clock::now();
        round++;
        vocab_capacity--;
    }
    ELOGFMT(WARN, "total train time: {}us", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count());
}

absl::flat_hash_map<size_t, std::string> BPETrainer::vocab() const {
    absl::flat_hash_map<size_t, std::string> r{};
    size_t offset{0};
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
    for (const auto& m : merges_) {
        r.emplace_back(std::string(m.first.value()), std::string(m.second.value()));
    }
    return r;
}

}  // namespace Tokenizer
