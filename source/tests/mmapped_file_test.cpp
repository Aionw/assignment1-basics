#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "mmapped_file.h"

namespace Tokenizer {

class MmappedFileTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MmappedFileTest, DefaultConstructor) {
    MmappedFile file;
    EXPECT_FALSE(file.isOpen());
    EXPECT_EQ(file.size(), 0);
}

TEST_F(MmappedFileTest, OpenNonExistentFile) {
    MmappedFile file;
    auto result = file.open("/nonexistent/path/file.txt", MmapAccess::read);
    EXPECT_FALSE(result.has_value());
    EXPECT_FALSE(file.isOpen());
}

TEST_F(MmappedFileTest, OpenExistingFile) {
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "mmapped_file_test.bin";
    std::ofstream ofs(tempFile, std::ios::binary);
    ofs << "test content";
    ofs.close();

    MmappedFile file(tempFile, MmapAccess::read);
    EXPECT_TRUE(file.isOpen());
    EXPECT_EQ(file.size(), 12);
    EXPECT_EQ(file.data(), "test content");

    std::filesystem::remove(tempFile);
}

TEST_F(MmappedFileTest, EmptyFile) {
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "mmapped_file_empty_test.bin";
    std::ofstream ofs(tempFile);
    ofs.close();

    MmappedFile file(tempFile, MmapAccess::read);
    EXPECT_TRUE(file.isOpen());
    EXPECT_EQ(file.size(), 0);

    std::filesystem::remove(tempFile);
}

TEST_F(MmappedFileTest, MoveConstructor) {
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "mmapped_file_move_test.bin";
    std::ofstream ofs(tempFile);
    ofs << "hello";
    ofs.close();

    MmappedFile file1(tempFile, MmapAccess::read);
    MmappedFile file2(std::move(file1));

    EXPECT_FALSE(file1.isOpen());
    EXPECT_TRUE(file2.isOpen());
    EXPECT_EQ(file2.size(), 5);
    EXPECT_EQ(file2.data(), "hello");

    std::filesystem::remove(tempFile);
}

TEST_F(MmappedFileTest, MoveAssignment) {
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "mmapped_file_move_assign_test.bin";
    std::ofstream ofs(tempFile);
    ofs << "world";
    ofs.close();

    MmappedFile file1(tempFile, MmapAccess::read);
    MmappedFile file2;
    file2 = std::move(file1);

    EXPECT_FALSE(file1.isOpen());
    EXPECT_TRUE(file2.isOpen());
    EXPECT_EQ(file2.size(), 5);
    EXPECT_EQ(file2.data(), "world");

    std::filesystem::remove(tempFile);
}

TEST_F(MmappedFileTest, CloseAndReopen) {
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "mmapped_file_reopen_test.bin";
    std::ofstream ofs(tempFile);
    ofs << "first";
    ofs.close();

    MmappedFile file;
    auto result1 = file.open(tempFile, MmapAccess::read);
    EXPECT_TRUE(result1.has_value());
    EXPECT_EQ(file.size(), 5);

    file.close();
    EXPECT_FALSE(file.isOpen());

    std::filesystem::path tempFile2 = std::filesystem::temp_directory_path() / "mmapped_file_reopen_test2.bin";
    std::ofstream ofs2(tempFile2);
    ofs2 << "second";
    ofs2.close();

    auto result2 = file.open(tempFile2, MmapAccess::read);
    EXPECT_TRUE(result2.has_value());
    EXPECT_TRUE(file.isOpen());
    EXPECT_EQ(file.size(), 6);
    EXPECT_EQ(file.data(), "second");

    std::filesystem::remove(tempFile);
    std::filesystem::remove(tempFile2);
}

}  // namespace Tokenizer
