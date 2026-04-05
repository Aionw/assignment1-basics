#include "pre_tokenizer.h"

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
#include "ylt/easylog.hpp"
#include "ylt/util/expected.hpp"

namespace Tokenizer {

//    // re2 regex usage
//    RE2 whitespace_re("\\s+");
//    if (!whitespace_re.ok()) {
//        return ylt::unexpected<std::string>("failed to compile regex");
//    }
//
//    std::string test_str = "hello   world";
//    std::string normalized = test_str;
//    RE2::GlobalReplace(&normalized, whitespace_re, " ");
//    ELOGFMT(DEBUG, "normalized: {}", normalized);
//
//    // absl Status/StatusOr usage
//    auto status = [](int64_t n) -> absl::StatusOr<int64_t> {
//        if (n < 0) {
//            return absl::InvalidArgumentError("negative not allowed");
//        }
//        return n * 2;
//    };
//
//    auto result = status(42);
//    if (result.ok()) {
//        ELOGFMT(DEBUG, "status_or result: {}", *result);
//    }
//
//    // absl Cleanup usage
//    bool cleaned_up = false;
//    {
//        auto cleanup = absl::MakeCleanup([&cleaned_up]() { cleaned_up = true; });
//        // cleanup runs when exiting this block
//    }
//    ELOGFMT(DEBUG, "cleanup fired: {}", cleaned_up ? "yes" : "no");
//
//    ELOGFMT(DEBUG, "read file content from: {} ok", file_path_.c_str());

ylt::expected<absl::flat_hash_map<std::string_view, size_t>, std::string> PreTokenizer::tokenize(
    MmappedFile& file) {
    if (!re_) {
        return ylt::unexpected<std::string>(re_.error());
    }

    std::vector<std::string_view> input_items{file.data()};
    std::vector<std::string_view> output_items{};
    if (special_tokens_.empty()) {
        output_items = input_items;
    } else {
        for (auto sep : special_tokens_) {
            for (auto item : input_items) {
                std::vector<std::string_view> splitted = absl::StrSplit(item, sep);
                output_items.reserve(splitted.size());
                output_items.insert(output_items.end(), splitted.begin(), splitted.end());
            }
            input_items = output_items;
        }
    }

    absl::flat_hash_map<std::string_view, size_t> token_count{};
    for (int i = 0; i < output_items.size(); ++i) {
        std::string_view item = output_items[i];
        re_.findAll(item, [&token_count](std::string_view match) { token_count[match]++; });
    }
    // NOTE: using another cv to wait token is better
    return {std::move(token_count)};
}

}  // namespace Tokenizer
