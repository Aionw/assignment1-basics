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

std::optional<size_t> WordItems::find(const WordItem& item) const {
    for (size_t i = 0; i < items_.size(); ++i) {
        if (items_[i] == item) {
            return i;
        }
    }
    return std::nullopt;
}

std::optional<size_t> WordItems::findPair(const WordItemPair& pair) const {
    auto left_index = find(pair.first);
    if (left_index && *left_index < items_.size() - 1 && items_[*left_index + 1] == pair.second) {
        return left_index;
    }
    return std::nullopt;
}

std::pair<std::unique_ptr<WordItemPair>, std::unique_ptr<WordItemPair>> WordItems::siblingPair(
    const WordItemPair& pair) const {
    std::optional<size_t> left_index = findPair(pair);
    if (!left_index) {
        return {};
    }

    std::unique_ptr<WordItemPair> left_pair{};
    std::unique_ptr<WordItemPair> right_pair{};
    if (left_index.value() != 0) {
        left_pair = std::make_unique<WordItemPair>(items_[left_index.value() - 1],
                                                   items_[left_index.value()]);
    }
    size_t right_index = *left_index + 1;
    if (right_index != items_.size() - 1) {
        right_pair = std::make_unique<WordItemPair>(items_[right_index], items_[right_index + 1]);
    }
    return {std::move(left_pair), std::move(right_pair)};
}

std::pair<std::unique_ptr<WordItemPair>, std::unique_ptr<WordItemPair>> WordItems::siblingPair(
    const WordItem& item) const {
    std::optional<size_t> index = find(item);
    if (!index) {
        return {nullptr, nullptr};
    }
    std::unique_ptr<WordItemPair> left_pair{};
    std::unique_ptr<WordItemPair> right_pair{};
    if (index.value() != 0) {
        left_pair =
            std::make_unique<WordItemPair>(items_[index.value() - 1], items_[index.value()]);
    }
    if (index.value() != items_.size() - 1) {
        right_pair =
            std::make_unique<WordItemPair>(items_[index.value()], items_[index.value() + 1]);
    }
    return {std::move(left_pair), std::move(right_pair)};
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

    auto pre_tokens = pre_tokenizer_.tokenize(file_);
    if (!pre_tokens.has_value()) {
        return;
    }

    absl::flat_hash_map<std::string_view, size_t> word_count = pre_tokens.value();
    absl::flat_hash_map<std::string_view, WordItems> word_items{};
    for (const auto& [k, _] : word_count) {
        word_items.emplace(k, k);
    }

    size_t vocab_capacity = vocab_size_ - pre_tokenizer_.getSpecialTokens().size() - 256;
    size_t total_merges = vocab_capacity;
    size_t current_merge = 0;
    size_t round{0};
    while (vocab_capacity != 0) {
        absl::flat_hash_map<PairType, size_t, WordItemPairHash> pair_count{};
        for (const auto& [k, items] : word_items) {
            for (const auto& p : items.pairs()) {
                pair_count[p] += word_count[k];
            }
        }
        if (pair_count.empty()) {
            break;
        }

        TopK<PairType, PairCountCompare> top_pairs(vocab_capacity,
                                                   PairCountCompare{.pair_count = pair_count});
        for (const auto& [p, _] : pair_count) {
            top_pairs.push(p);
        }

        PairType top_pair = top_pairs.get()[0];
        auto sort_pairs = top_pairs.get_sorted();
        for (auto it = sort_pairs.rbegin(); it != sort_pairs.rend(); ++it) {
            ELOGFMT(WARN, "R{} p({}:{}) count: {}", round, it->first.value(), it->second.value(),
                    pair_count[*it]);
        }
        round++;
        merges_.emplace_back(top_pair);

        for (auto& [k, items] : word_items) {
            items.merge(top_pair);
        }
        vocab_capacity--;
    }
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
