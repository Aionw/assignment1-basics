#include "mmapped_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>

namespace Tokenizer {

namespace {

int toOpenFlags(MmapAccess access) {
    switch (access) {
        case MmapAccess::read:
            return O_RDONLY;
        case MmapAccess::write:
            return O_WRONLY;
        case MmapAccess::read_write:
            return O_RDWR;
    }
    return O_RDONLY;
}

int toProtFlags(MmapAccess access) {
    switch (access) {
        case MmapAccess::read:
            return PROT_READ;
        case MmapAccess::write:
            return PROT_WRITE;
        case MmapAccess::read_write:
            return PROT_READ | PROT_WRITE;
    }
    return PROT_READ;
}

}  // namespace

MmappedFile::MmappedFile(const std::filesystem::path& path, MmapAccess access) : path_(path) {
    open(path, access);
}

MmappedFile::~MmappedFile() { close(); }

MmappedFile::MmappedFile(MmappedFile&& other) noexcept
    : path_(other.path_),
      addr_(other.addr_),
      size_(other.size_),
      fd_(other.fd_),
      access_(other.access_) {
    other.addr_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
}

MmappedFile& MmappedFile::operator=(MmappedFile&& other) noexcept {
    if (this != &other) {
        close();
        path_ = std::move(other.path_);
        addr_ = other.addr_;
        size_ = other.size_;
        fd_ = other.fd_;
        access_ = other.access_;
        other.addr_ = nullptr;
        other.size_ = 0;
        other.fd_ = -1;
    }
    return *this;
}

ylt::expected<void, std::string> MmappedFile::open(const std::filesystem::path& path,
                                                   MmapAccess access) {
    close();
    path_ = path;

    fd_ = ::open(path.c_str(), toOpenFlags(access));
    if (fd_ == -1) {
        return ylt::unexpected<std::string>(std::string("failed to open file: ") +
                                            std::strerror(errno));
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        ::close(fd_);
        fd_ = -1;
        return ylt::unexpected<std::string>(std::string("failed to stat file: ") +
                                            std::strerror(errno));
    }

    size_ = sb.st_size;
    access_ = access;

    if (size_ == 0) {
        ELOGFMT(DEBUG, "mmap file: {} (empty file)", path.c_str());
        return {};
    }

    addr_ = mmap(nullptr, size_, toProtFlags(access), MAP_PRIVATE, fd_, 0);
    if (addr_ == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        addr_ = nullptr;
        size_ = 0;
        return ylt::unexpected<std::string>(std::string("failed to mmap: ") + std::strerror(errno));
    }

    ELOGFMT(DEBUG, "mmap file: {} ({} bytes) ok", path.c_str(), size_);
    return {};
}

void MmappedFile::close() {
    if (addr_ != nullptr && addr_ != MAP_FAILED) {
        munmap(addr_, size_);
    }
    if (fd_ != -1) {
        ::close(fd_);
    }
    addr_ = nullptr;
    size_ = 0;
    fd_ = -1;
    path_.clear();
}

std::string_view MmappedFile::data() const {
    return std::string_view(static_cast<const char*>(addr_), size_);
}

size_t MmappedFile::size() const { return size_; }

}  // namespace Tokenizer
