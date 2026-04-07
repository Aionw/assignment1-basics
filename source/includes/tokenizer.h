#pragma once

#include <cstddef>
#include <cstring>
#include <algorithm>
#include <functional>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "mmapped_file.h"
#include "pre_tokenizer.h"
#include "thread_pool.h"
#include "ylt/easylog.hpp"

namespace Tokenizer {

template <typename T, typename Compare = std::less<T>>
class TopK {
public:
    TopK(size_t k, Compare comp = Compare{}) : k_(k), comp_(comp) {}

    void push(const T& val) {
        if (heap_.size() < k_) {
            heap_.push_back(val);
            std::push_heap(heap_.begin(), heap_.end(), comp_);
        } else if (!heap_.empty()) {
            auto min_it = std::min_element(heap_.begin(), heap_.end(), comp_);
            if (comp_(*min_it, val)) {
                *min_it = val;
            }
            std::make_heap(heap_.begin(), heap_.end(), comp_);
        }
    }

    const std::vector<T>& get() const { return heap_; }
    void clear() { heap_.clear(); }

    std::vector<T> get_sorted() const {
        std::vector<T> result = heap_;
        std::sort(result.begin(), result.end(), comp_);
        return result;
    }

private:
    size_t k_;
    Compare comp_;
    std::vector<T> heap_;
};

class WordItem {
public:
    WordItem() : owned_(false), data_(nullptr), offset_(0), size_(0) {}
    explicit WordItem(std::string_view sv) {
        owned_ = true;
        auto data = std::make_unique<char[]>(sv.size());
        ::memcpy(data.get(), sv.data(), sv.size());
        offset_ = 0;
        size_ = sv.size();
        data_ = data.release();
    }
    WordItem(const WordItem& other) { copy_from(other); }
    WordItem(const char* data, size_t offset, size_t sz, bool owned)
        : owned_(owned), data_(data), offset_(offset), size_(sz) {}
    WordItem(const char* data, size_t offset) : WordItem(data, offset, 1, false) {}
    ~WordItem() {
        if (owned_) {
            delete data_;
            data_ = nullptr;
        }
    }

    WordItem& operator=(const WordItem& other) {
        copy_from(other);
        return *this;
    }

    friend bool operator==(const WordItem& a, const WordItem& b) { return a.value() == b.value(); }

    friend absl::HashState AbslHashValue(absl::HashState state, const WordItem& item) {
        return absl::HashState::combine(std::move(state), item.value());
    }

    friend WordItem operator+(const WordItem& a, const WordItem& b) {
        if (!a.owned_ && !b.owned_ && a.data_ == b.data_ && a.offset_ + a.size_ == b.offset_) {
            return WordItem(a.data_, a.offset_, a.size_ + b.size_, false);
        }
        auto data = std::make_unique<char[]>(a.size_ + b.size_);
        ::memcpy(data.get(), a.data_ + a.offset_, a.size_);
        ::memcpy(data.get() + a.size_, b.data_ + b.offset_, b.size_);
        return WordItem(data.release(), 0, a.size_ + b.size_, true);
    }

    std::string_view value() const { return std::string_view(data_ + offset_, size_); }
    void copy_from(const WordItem& other) {
        reset();
        if (other.owned_) {
            owned_ = true;
            auto data = std::make_unique<char[]>(other.size_);
            ::memcpy(data.get(), other.data_ + other.offset_, other.size_);
            data_ = data.release();
            offset_ = 0;
            size_ = other.size_;
        } else {
            owned_ = false;
            data_ = other.data_;
            offset_ = other.offset_;
            size_ = other.size_;
        }
    }
    void move_from(WordItem&& other) {
        reset();
        owned_ = other.owned_;
        data_ = other.data_;
        offset_ = other.offset_;
        size_ = other.size_;
        other.reset();
    }

private:
    void reset() {
        if (owned_ && data_ != nullptr) {
            delete[] data_;
        }
        owned_ = false;
        data_ = nullptr;
        offset_ = 0;
        size_ = 0;
    }

    bool owned_{false};
    const char* data_;
    size_t offset_;
    size_t size_;
};

using WordItemPair = std::pair<WordItem, WordItem>;

struct WordItemPairHash {
    size_t operator()(const WordItemPair& pair) const {
        auto h1 = absl::Hash<std::string_view>{}(pair.first.value());
        auto h2 = absl::Hash<std::string_view>{}(pair.second.value());
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct WordItemHash {
    size_t operator()(const WordItem& item) const {
        return absl::Hash<std::string_view>{}(item.value());
    }
};

struct PairCountCompare {
    const absl::flat_hash_map<std::pair<WordItem, WordItem>, size_t, WordItemPairHash>&
        pair_count{};

    bool operator()(const std::pair<WordItem, WordItem>& a,
                    const std::pair<WordItem, WordItem>& b) const {
        size_t a_count = pair_count.contains(a) ? pair_count.at(a) : 0;
        size_t b_count = pair_count.contains(b) ? pair_count.at(b) : 0;
        if (a_count == b_count && a.first == b.first) {
            return a.second.value() < b.second.value();
        } else if (a_count == b_count) {
            return a.first.value() < b.first.value();
        }
        return a_count < b_count;
    }
};

class WordItems {
public:
    explicit WordItems(std::string_view word);

    std::vector<WordItemPair> pairs() const;
    const std::vector<WordItem>& items() const { return items_; }

    std::pair<std::unique_ptr<WordItemPair>, std::unique_ptr<WordItemPair>> siblingPair(
        const WordItemPair& pair) const;
    std::pair<std::unique_ptr<WordItemPair>, std::unique_ptr<WordItemPair>> siblingPair(
        const WordItem& item) const;
    void merge(const WordItemPair& pair);

private:
    std::optional<size_t> find(const WordItem& item) const;
    std::optional<size_t> findPair(const WordItemPair& pair) const;

    std::vector<WordItem> items_;
};

class BPETrainer {
    static constexpr std::string_view kGPT2Pattern =
        "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

public:
    BPETrainer(size_t thread_count, const std::string& file_path,
               const std::vector<std::string>& special_tokens, size_t vocab_size);

    void train();

    absl::flat_hash_map<size_t, std::string> vocab() const;
    std::vector<std::pair<std::string, std::string>> merges() const;

private:
    ThreadPool pool_;
    MmappedFile file_;
    size_t vocab_size_;
    PreTokenizer pre_tokenizer_;
    std::vector<std::pair<WordItem, WordItem>> merges_{};
};

}  // namespace Tokenizer
