#include <cstddef>
#include <cstdio>

#include <gtest/gtest.h>
#include <fstream>
#include "absl/container/flat_hash_map.h"

#include "bpe_trainer.h"

namespace Tokenizer {

class BPETrainerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_file_path_ = "/tmp/bpe_test_input.txt";
        std::ofstream file(test_file_path_);
        file << "hello world";
        file.close();
    }

    void TearDown() override {
        std::remove(test_file_path_.c_str());
    }

    std::string test_file_path_;
};

TEST_F(BPETrainerTest, Constructor) {
    BPETrainer trainer(1, test_file_path_, {"<unk>"}, 300);
    EXPECT_TRUE(trainer.merges().empty());
}

TEST_F(BPETrainerTest, Train) {
    BPETrainer trainer(1, test_file_path_, {"<unk>"}, 300);
    trainer.train();
    EXPECT_FALSE(trainer.merges().empty());
}

TEST_F(BPETrainerTest, VocabAfterTrain) {
    BPETrainer trainer(1, test_file_path_, {"<unk>"}, 300);
    trainer.train();
    auto vocab = trainer.vocab();
    EXPECT_GE(vocab.size(), 257);
}

TEST_F(BPETrainerTest, VocabContainsSpecialTokens) {
    BPETrainer trainer(1, test_file_path_, {"<unk>", "<pad>"}, 300);
    trainer.train();
    auto vocab = trainer.vocab();
    EXPECT_EQ(vocab[0], "<unk>");
    EXPECT_EQ(vocab[1], "<pad>");
}

TEST_F(BPETrainerTest, MergesArePairsOfStrings) {
    BPETrainer trainer(1, test_file_path_, {}, 300);
    trainer.train();
    auto merges = trainer.merges();
    for (const auto& [left, right] : merges) {
        EXPECT_FALSE(left.empty());
        EXPECT_FALSE(right.empty());
    }
}

TEST_F(BPETrainerTest, EmptyFile) {
    std::string empty_file = "/tmp/bpe_empty_test.txt";
    std::ofstream file(empty_file);
    file.close();

    BPETrainer trainer(1, empty_file, {}, 300);
    trainer.train();
    auto merges = trainer.merges();
    EXPECT_TRUE(merges.empty());

    std::remove(empty_file.c_str());
}

TEST_F(BPETrainerTest, VocabSizeMatchesSpec) {
    std::string file_path = "/tmp/bpe_size_test.txt";
    std::ofstream file(file_path);
    file << "aa bb cc dd";
    file.close();

    size_t vocab_size = 300;
    size_t special_count = 2;
    BPETrainer trainer(1, file_path, {"<unk>", "<pad>"}, vocab_size);
    trainer.train();
    auto vocab = trainer.vocab();
    EXPECT_EQ(vocab.size(), special_count + 256 + trainer.merges().size());

    std::remove(file_path.c_str());
}

class TopKTest : public ::testing::Test {};

TEST_F(TopKTest, KeepsTop3Largest) {
    TopK<int> topk(3);
    std::vector<int> vals = {1, 5, 3, 2, 4};
    for (int v : vals) {
        topk.push(v);
    }
    auto result = topk.get();
    // Print for debugging
    for (size_t i = 0; i < result.size(); ++i) {
        fprintf(stderr, "result[%zu] = %d\n", i, result[i]);
    }
    std::sort(result.begin(), result.end(), std::greater<int>());
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 5);
    EXPECT_EQ(result[1], 4);
    EXPECT_EQ(result[2], 3);
}

TEST_F(TopKTest, KEqualsOne) {
    TopK<int> topk(1);
    topk.push(5);
    topk.push(10);
    topk.push(3);
    auto result = topk.get();
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 10);
}

TEST_F(TopKTest, FewerThanKElements) {
    TopK<int> topk(5);
    topk.push(3);
    topk.push(1);
    auto result = topk.get();
    EXPECT_EQ(result.size(), 2);
}

TEST_F(TopKTest, Empty) {
    TopK<int> topk(3);
    auto result = topk.get();
    EXPECT_TRUE(result.empty());
}

TEST_F(TopKTest, ReplaceSmaller) {
    TopK<int> topk(2);
    topk.push(10);
    topk.push(20);
    topk.push(5);
    auto result = topk.get();
    EXPECT_EQ(result.size(), 2);
    EXPECT_GE(result[0], 5);
    EXPECT_GE(result[1], 5);
}

TEST_F(TopKTest, SortedReturnsMaxFirst) {
    TopK<int> topk(3);
    topk.push(1);
    topk.push(5);
    topk.push(3);
    topk.push(2);
    topk.push(4);
    auto sorted = topk.get_sorted();
    EXPECT_EQ(sorted.back(), 5);
}

TEST_F(TopKTest, WordTest) {
    absl::flat_hash_map<WordItemPair, size_t, WordItemPairHash> pair_counts{
    };
    pair_counts.emplace(WordItemPair(WordItem("a"), WordItem("b")), 10);
    pair_counts.emplace(WordItemPair(WordItem("a"), WordItem("c")), 10);
    pair_counts.emplace(WordItemPair(WordItem("b"), WordItem("zz")), 10);
    pair_counts.emplace(WordItemPair(WordItem("ba"), WordItem("a")), 10);
    TopK<WordItemPair, PairCountCompare> topk(3, PairCountCompare{pair_counts});
    for (const auto& [p, _] : pair_counts) {
        topk.push(p);
    }
    EXPECT_EQ(topk.get()[0] ,WordItemPair(WordItem("ba"), WordItem("a")));
}

class WordItemTest: public ::testing::Test {};

TEST_F(WordItemTest, TestMerge) {
    WordItems items("xxxxxx");
    EXPECT_EQ(items.pairs().size(), 5);
    items.merge(WordItemPair("x", "x"));
    EXPECT_EQ(items.pairs().size(), 2);

    WordItems items0("axab");
    EXPECT_EQ(items0.pairs().size(), 3);
    items0.merge(WordItemPair("a", "b"));
    EXPECT_EQ(items0.pairs().size(), 2);


    WordItems items1("abc");
    EXPECT_EQ(items1.pairs().size(), 2);
    items1.merge(WordItemPair("a", "b"));
    EXPECT_EQ(items1.pairs().size(), 1);
}

}  // namespace Tokenizer
