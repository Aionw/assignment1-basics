#include "pre_tokenizer.h"
#include "ylt/easylog.hpp"
#include <CLI/CLI.hpp>
#include <vector>
#include <string>
#include "mmapped_file.h"

int main(int argc, char** argv) {
    CLI::App app{"PreTokenizer"};

    std::string file_path;
    std::vector<std::string> special_tokens;

    app.add_option("-i,--input", file_path, "Input file path")
        ->required(true);
    app.add_option("-s,--special-token", special_tokens, "Special tokens");

    CLI11_PARSE(app, argc, argv);

    const char* pattern =
        "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    Tokenizer::PreTokenizer tokenizer(pattern, special_tokens);
    Tokenizer::MmappedFile mm_file{file_path, Tokenizer::MmapAccess::read};
    auto result = tokenizer.tokenize(mm_file);

    if (!result) {
        ELOGFMT(ERROR, "Tokenization failed: {}", result.error());
        return 1;
    }
    ELOGFMT(WARN, "got pre-token result: {}", result.value());

    return 0;
}
