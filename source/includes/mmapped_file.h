#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "ylt/easylog.hpp"
#include "ylt/util/expected.hpp"

namespace Tokenizer {

enum class MmapAccess { read, write, read_write };

class MmappedFile {
public:
    MmappedFile() = default;
    MmappedFile(const std::filesystem::path& path, MmapAccess access);
    ~MmappedFile();

    MmappedFile(const MmappedFile&) = delete;
    MmappedFile& operator=(const MmappedFile&) = delete;

    MmappedFile(MmappedFile&& other) noexcept;
    MmappedFile& operator=(MmappedFile&& other) noexcept;

    ylt::expected<void, std::string> open(const std::filesystem::path& path, MmapAccess access);
    void close();

    std::string_view data() const;
    size_t size() const;
    bool isOpen() const { return fd_ != -1; }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    void* addr_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;
    MmapAccess access_ = MmapAccess::read;
};

}  // namespace Tokenizer
