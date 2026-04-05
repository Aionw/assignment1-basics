#include "pcre2_regex.h"

#include "fmt/format.h"

#include <algorithm>

namespace Tokenizer {

Pcre2Regex::~Pcre2Regex() = default;

Pcre2Regex::Pcre2Regex(const char* pattern, uint32_t options) {
    compile(pattern, options);
}

Pcre2Regex::Pcre2Regex(Pcre2Regex&& other) noexcept
    : code_(std::move(other.code_)),
      error_code_(other.error_code_),
      error_message_(std::move(other.error_message_)) {
    other.error_code_ = 0;
}

Pcre2Regex& Pcre2Regex::operator=(Pcre2Regex&& other) noexcept {
    if (this != &other) {
        code_ = std::move(other.code_);
        error_code_ = other.error_code_;
        error_message_ = std::move(other.error_message_);
        other.error_code_ = 0;
    }
    return *this;
}

bool Pcre2Regex::compile(const char* pattern, uint32_t options) {
    code_.reset();
    error_message_.clear();
    error_code_ = 0;

    PCRE2_SIZE erroffset;
    pcre2_code* re = pcre2_compile((PCRE2_SPTR)pattern, PCRE2_ZERO_TERMINATED,
                                     options, &error_code_, &erroffset, nullptr);
    if (!re) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(error_code_, buffer, sizeof(buffer));
        error_message_ = fmt::format("failed to compile regex at offset {}: {}",
                                     (size_t)erroffset, (char*)buffer);
        return false;
    }

    code_.reset(re);
    pcre2_jit_compile(re, PCRE2_JIT_COMPLETE);
    return true;
}

std::vector<std::string_view> Pcre2Regex::findAll(std::string_view text) const {
    std::vector<std::string_view> results;
    findAll(text, [&results](std::string_view match) {
        results.push_back(match);
    });
    return results;
}

void Pcre2Regex::findAll(std::string_view text,
                         const std::function<void(std::string_view)>& callback) const {
    if (!code_) {
        return;
    }

    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(code_.get(), nullptr);
    if (!match_data) {
        return;
    }

    PCRE2_SIZE subject_len = text.size();
    PCRE2_SIZE offset = 0;

    while (pcre2_match(code_.get(), (PCRE2_SPTR)text.data(), subject_len, offset,
                       0, match_data, nullptr) >= 0) {
        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE match_start = ovector[0];
        PCRE2_SIZE match_end = ovector[1];
        callback(text.substr(match_start, match_end - match_start));
        offset = match_end;
    }

    pcre2_match_data_free(match_data);
}

}  // namespace Tokenizer
