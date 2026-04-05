#pragma once

#define PCRE2_CODE_UNIT_WIDTH 8
#include "pcre2.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace Tokenizer {

class Pcre2Regex {
public:
    Pcre2Regex() = default;
    explicit Pcre2Regex(const char* pattern, uint32_t options = PCRE2_UTF);

    ~Pcre2Regex();

    Pcre2Regex(const Pcre2Regex&) = delete;
    Pcre2Regex& operator=(const Pcre2Regex&) = delete;

    Pcre2Regex(Pcre2Regex&& other) noexcept;
    Pcre2Regex& operator=(Pcre2Regex&& other) noexcept;

    bool compile(const char* pattern, uint32_t options = PCRE2_UTF);
    bool ok() const { return code_ != nullptr; }
    explicit operator bool() const { return ok(); }

    std::vector<std::string_view> findAll(std::string_view text) const;
    void findAll(std::string_view text, const std::function<void(std::string_view)>& callback) const;

    const char* error() const { return error_message_.c_str(); }
    int errorCode() const { return error_code_; }

private:
    struct CodeDeleter {
        void operator()(pcre2_code* code) const {
            if (code) {
                pcre2_code_free(code);
            }
        }
    };

    std::unique_ptr<pcre2_code, CodeDeleter> code_;
    int error_code_ = 0;
    std::string error_message_;
};

}  // namespace Tokenizer
