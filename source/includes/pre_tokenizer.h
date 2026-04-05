#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "fmt/ranges.h"
#include "mmapped_file.h"
#include "pcre2_regex.h"
#include "thread_pool.h"
#include "ylt/easylog.hpp"
#include "ylt/util/expected.hpp"

namespace Tokenizer {

class PreTokenizer {
public:
    PreTokenizer(ThreadPool& pool, const char* token_pattern,
                 const std::vector<std::string>& special_tokens)
        : pool_(pool), re_(token_pattern), special_tokens_(special_tokens) {}

    ylt::expected<absl::flat_hash_map<std::string_view, size_t>, std::string> tokenize(MmappedFile& file);

    const std::vector<std::string>& getSpecialTokens() const { return special_tokens_; }

private:
    ThreadPool& pool_;
    Pcre2Regex re_;
    std::vector<std::string> special_tokens_{};
};

}  // namespace Tokenizer
