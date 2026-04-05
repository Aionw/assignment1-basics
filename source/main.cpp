#include "pre_tokenizer.h"
#include "thread_pool.h"
#include "ylt/easylog.hpp"
#include <CLI/CLI.hpp>
#include <vector>
#include <string>
#include "mmapped_file.h"

int main(int argc, char** argv) {
    CLI::App app{"PreTokenizer"};

    std::string file_path;
    std::vector<std::string> special_tokens;
    size_t thread_count = 1;

    app.add_option("-i,--input", file_path, "Input file path")
        ->required(true);
    app.add_option("-s,--special-token", special_tokens, "Special tokens");
    app.add_option("-t,--threads", thread_count, "Thread pool size")
        ->default_val(1);

    CLI11_PARSE(app, argc, argv);

    Tokenizer::ThreadPool pool(thread_count);
    const char* pattern =
        "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    Tokenizer::PreTokenizer tokenizer(pool, pattern, special_tokens);
    Tokenizer::MmappedFile mm_file{file_path, Tokenizer::MmapAccess::read};
    auto result = tokenizer.tokenize(mm_file);

    if (!result) {
        ELOGFMT(ERROR, "Tokenization failed: {}", result.error());
        return 1;
    }

    return 0;
}
