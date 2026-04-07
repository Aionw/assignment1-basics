cmake_minimum_required(VERSION 3.10)

# Set Python and include path BEFORE project() so all targets see it
set(Python_EXECUTABLE /opt/homebrew/bin/python3.13 CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13")

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
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.12.0
)
FetchContent_Declare(
  abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20240722.0
)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
FetchContent_Declare(
  pcre2
  GIT_REPOSITORY https://github.com/PhilipHazel/pcre2.git
  GIT_TAG pcre2-10.44
)
FetchContent_MakeAvailable(yalantinglibs cli11 fmt nanobind abseil googletest pcre2)

include_directories(${CMAKE_SOURCE_DIR}/includes)
include_directories(${CMAKE_SOURCE_DIR}/build/_deps/yalantinglibs-src/include)
include_directories(${CMAKE_SOURCE_DIR}/build/_deps/abseil-src)

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

add_library(tokenizer src/tokenizer.cpp)
target_link_libraries(tokenizer 
  yalantinglibs 
  absl::flat_hash_map
  mmapped_file
  pre_tokenizer
  thread_pool
)

nanobind_add_module(ctokenizer bindings/bindings.cpp)
target_link_libraries(ctokenizer PRIVATE tokenizer)
target_include_directories(ctokenizer PRIVATE ${CMAKE_SOURCE_DIR}/includes)
target_include_directories(ctokenizer PRIVATE /opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13)

add_executable(tokenizer_test tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_test tokenizer pre_tokenizer mmapped_file pcre2_regex thread_pool GTest::gtest_main GTest::gtest yalantinglibs fmt absl::flat_hash_map absl::flat_hash_set absl::strings)
add_test(NAME tokenizer_test COMMAND tokenizer_test)

add_executable(myapp main.cpp)
target_link_libraries(myapp pre_tokenizer CLI11 fmt)

enable_testing()
