#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#include "mmapped_file.h"
#include "pcre2_regex.h"
#include "pre_tokenizer.h"
#include "thread_pool.h"

namespace Tokenizer {

class PreTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_unique<ThreadPool>(4);
    }

    void TearDown() override {
        pool_.reset();
    }

    std::unique_ptr<ThreadPool> pool_;
};

TEST_F(PreTokenizerTest, Constructor) {
    const char* pattern = "'t|est";
    std::vector<std::string> tokens = {"<unk>", "<pad>"};
    PreTokenizer tokenizer(*pool_, pattern, tokens);
    EXPECT_EQ(tokenizer.getSpecialTokens(), tokens);
}

TEST_F(PreTokenizerTest, EmptySpecialTokens) {
    const char* pattern = "test";
    std::vector<std::string> tokens;
    PreTokenizer tokenizer(*pool_, pattern, tokens);
    EXPECT_TRUE(tokenizer.getSpecialTokens().empty());
}

TEST_F(PreTokenizerTest, TokenizeSimpleText) {
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "pretokenizer_test.txt";
    std::ofstream ofs(tempFile);
    ofs << "hello";
    ofs.close();

    const char* pattern = "[a-zA-Z]+";
    std::vector<std::string> tokens;
    PreTokenizer tokenizer(*pool_, pattern, tokens);

    MmappedFile file(tempFile, MmapAccess::read);
    ASSERT_TRUE(file.isOpen());

    auto result = tokenizer.tokenize(file);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value().size(), 1);

    std::filesystem::remove(tempFile);
}

TEST_F(PreTokenizerTest, TokenizeWithSpecialTokens) {
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "pretokenizer_test.txt";
    std::ofstream ofs(tempFile);
    ofs << "hello<|endoftext|>world<|endoftext|>!";
    ofs.close();

    const char* pattern = "[a-z]+";
    std::vector<std::string> tokens = {"<|endoftext|>"};
    PreTokenizer tokenizer(*pool_, pattern, tokens);

    MmappedFile file(tempFile, MmapAccess::read);
    ASSERT_TRUE(file.isOpen());

    auto result = tokenizer.tokenize(file);
    EXPECT_TRUE(result.has_value());

    std::filesystem::remove(tempFile);
}

TEST_F(PreTokenizerTest, TokenizeEmptyFile) {
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "pretokenizer_test.txt";
    std::ofstream ofs(tempFile);
    ofs << "";
    ofs.close();

    const char* pattern = "[a-z]+";
    std::vector<std::string> tokens;
    PreTokenizer tokenizer(*pool_, pattern, tokens);

    MmappedFile file(tempFile, MmapAccess::read);
    ASSERT_TRUE(file.isOpen());

    auto result = tokenizer.tokenize(file);
    EXPECT_TRUE(result.has_value());

    std::filesystem::remove(tempFile);
}

TEST_F(PreTokenizerTest, TokenizeCounts) {
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "pretokenizer_test.txt";
    std::ofstream ofs(tempFile);
    ofs << "aaa bbb";
    ofs.close();

    const char* pattern = "[a-z]+";
    std::vector<std::string> tokens;
    PreTokenizer tokenizer(*pool_, pattern, tokens);

    MmappedFile file(tempFile, MmapAccess::read);
    ASSERT_TRUE(file.isOpen());

    auto result = tokenizer.tokenize(file);
    EXPECT_TRUE(result.has_value());

    auto& counts = result.value();
    EXPECT_EQ(counts["aaa"], 1);
    EXPECT_EQ(counts["bbb"], 1);

    std::filesystem::remove(tempFile);
}

TEST_F(PreTokenizerTest, TokenizeWithUnicode) {
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "pretokenizer_test.txt";
    std::ofstream ofs(tempFile);
    ofs << "hello 世界 123";
    ofs.close();

    const char* pattern = "\\p{L}+|\\p{N}+";
    std::vector<std::string> tokens;
    PreTokenizer tokenizer(*pool_, pattern, tokens);

    MmappedFile file(tempFile, MmapAccess::read);
    ASSERT_TRUE(file.isOpen());

    auto result = tokenizer.tokenize(file);
    EXPECT_TRUE(result.has_value());

    std::filesystem::remove(tempFile);
}

}  // namespace Tokenizer
