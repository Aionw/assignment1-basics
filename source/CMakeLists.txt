cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
  yalantinglibs
  GIT_REPOSITORY https://github.com/alibaba/yalantinglibs.git
  GIT_TAG 0.6.0
)
FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.6.2
)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.0.2
)
FetchContent_MakeAvailable(yalantinglibs cli11 fmt)

# Use Homebrew Abseil
find_package(absl REQUIRED)

# Use Homebrew GTest
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  # Create imported targets for Homebrew GTest
  if(NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main STATIC IMPORTED)
    set_target_properties(GTest::gtest_main PROPERTIES
      IMPORTED_LOCATION /opt/homebrew/lib/libgtest_main.a
    )
  endif()
  if(NOT TARGET GTest::gtest)
    add_library(GTest::gtest STATIC IMPORTED)
    set_target_properties(GTest::gtest PROPERTIES
      IMPORTED_LOCATION /opt/homebrew/lib/libgtest.a
    )
  endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/includes)
include_directories(/opt/homebrew/opt/pcre2/include)
link_directories(/opt/homebrew/opt/pcre2/lib)
include_directories(/opt/homebrew/opt/abseil/include)

add_library(mmapped_file src/mmapped_file.cpp)
target_link_libraries(mmapped_file yalantinglibs)

add_library(thread_pool src/thread_pool.cpp)

add_library(pcre2_regex src/pcre2_regex.cpp)
target_link_libraries(pcre2_regex pcre2-8 fmt)

add_library(pre_tokenizer src/pre_tokenizer.cpp)
target_link_libraries(
  pre_tokenizer
  yalantinglibs
  fmt
  absl::strings
  absl::flat_hash_map
  mmapped_file
  thread_pool
  pcre2_regex
)

add_executable(pre_tokenizer_test tests/pre_tokenizer_test.cpp)
target_link_libraries(pre_tokenizer_test pre_tokenizer GTest::gtest_main GTest::gtest fmt)
add_test(NAME pre_tokenizer_test COMMAND pre_tokenizer_test)

add_executable(mmapped_file_test tests/mmapped_file_test.cpp)
target_link_libraries(mmapped_file_test mmapped_file GTest::gtest_main GTest::gtest fmt)
add_test(NAME mmapped_file_test COMMAND mmapped_file_test)

add_executable(myapp main.cpp)
target_link_libraries(myapp pre_tokenizer CLI11 fmt)

enable_testing()
