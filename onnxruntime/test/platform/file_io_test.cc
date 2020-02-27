// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"

#include <fstream>
#include <random>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <unistd.h>  // for sysconf() and _SC_PAGESIZE
#endif

#include "gsl/gsl"

#include "gtest/gtest.h"

#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"

namespace onnxruntime {
namespace test {

namespace {
struct TempFilePath {
  TempFilePath(const PathString& base)
      : path{[&base]() {
          PathString path_template = base + ORT_TSTR("XXXXXX");
          int fd;
          CreateTestFile(fd, path_template);
#ifdef _WIN32
          _close(fd);
#else
          close(fd);
#endif
          return path_template;
        }()} {
  }

  ~TempFilePath() {
    DeleteFileFromDisk(path.c_str());
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TempFilePath);

  const PathString path;
};

std::vector<char> GenerateData(size_t length, uint32_t seed = 0) {
  auto engine = std::default_random_engine{seed};
  auto dist = std::uniform_int_distribution<int>{
      std::numeric_limits<char>::min(), std::numeric_limits<char>::max()};
  std::vector<char> result{};
  result.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    result.push_back(static_cast<char>(dist(engine)));
  }
  return result;
}

Status WriteDataToFile(gsl::span<const char> data, const PathString& path) {
  std::ofstream out{path, std::ios::binary | std::ios::trunc};
  ORT_RETURN_IF_NOT(out);
  if (data.empty()) return Status::OK();
  ORT_RETURN_IF_NOT(out.write(data.data(), data.size()));
  return Status::OK();
}

Status ReadDataFromFile(const PathString& path, gsl::span<char> data) {
  std::ifstream in{path, std::ios::binary};
  ORT_RETURN_IF_NOT(in);
  if (data.empty()) return Status::OK();
  ORT_RETURN_IF_NOT(in.read(data.data(), data.size()));
  return Status::OK();
}

std::vector<std::pair<FileOffsetType, size_t>> GenerateValidOffsetLengthPairs(size_t begin, size_t end, size_t interval = 1) {
  std::vector<std::pair<FileOffsetType, size_t>> offset_length_pairs;
  for (size_t range_begin = begin; range_begin < end; range_begin += interval) {
    for (size_t range_end = range_begin; range_end <= end; range_end += interval) {
      offset_length_pairs.emplace_back(static_cast<FileOffsetType>(range_begin), range_end - range_begin);
    }
  }
  return offset_length_pairs;
}
}  // namespace

TEST(FileIoTest, ReadFileIntoBuffer) {
  TempFilePath tmp(ORT_TSTR("read_test_"));
  const auto expected_data = GenerateData(32);
  ASSERT_STATUS_OK(WriteDataToFile(gsl::make_span(expected_data), tmp.path));

  const auto offsets_and_lengths = GenerateValidOffsetLengthPairs(0, expected_data.size());

  std::vector<char> buffer(expected_data.size());
  for (const auto& offset_and_length : offsets_and_lengths) {
    const auto offset = offset_and_length.first;
    const auto length = offset_and_length.second;

    SCOPED_TRACE(MakeString("offset: ", offset, ", length: ", length));

    auto buffer_span = gsl::make_span(buffer.data(), length);
    ASSERT_STATUS_OK(Env::Default().ReadFileIntoBuffer(tmp.path.c_str(), offset, length, buffer_span));

    auto expected_data_span = gsl::make_span(expected_data.data() + offset, length);

    ASSERT_EQ(buffer_span, expected_data_span);
  }

  // invalid - negative offset
  ASSERT_FALSE(Env::Default().ReadFileIntoBuffer(tmp.path.c_str(), -1, 0, gsl::make_span(buffer)).IsOK());

  // invalid - length too long
  ASSERT_FALSE(Env::Default().ReadFileIntoBuffer(tmp.path.c_str(), 0, expected_data.size() + 1, gsl::make_span(buffer)).IsOK());

  // invalid - buffer too short
  ASSERT_FALSE(Env::Default().ReadFileIntoBuffer(tmp.path.c_str(), 0, 3, gsl::make_span(buffer.data(), 2)).IsOK());
}

TEST(FileIoTest, WriteBufferIntoFile) {
  TempFilePath tmp(ORT_TSTR("write_test_"));
  const auto buffer = GenerateData(32);
  const auto buffer_span = gsl::make_span(buffer);
  const auto zeros = std::vector<char>(buffer.size());
  const auto zeros_span = gsl::make_span(zeros);

  const auto offsets_and_lengths = GenerateValidOffsetLengthPairs(0, buffer.size());

  std::vector<char> read_buffer(buffer.size());
  const auto read_buffer_span = gsl::make_span(read_buffer);

  for (const auto& offset_and_length : offsets_and_lengths) {
    const auto offset = offset_and_length.first;
    const auto length = offset_and_length.second;

    SCOPED_TRACE(MakeString("offset: ", offset, ", length: ", length));

    // zero first
    ASSERT_STATUS_OK(WriteDataToFile(zeros_span, tmp.path));

    ASSERT_STATUS_OK(Env::Default().WriteBufferIntoFile(buffer_span, tmp.path.c_str(), offset, length));

    ASSERT_STATUS_OK(ReadDataFromFile(tmp.path, read_buffer_span));

    ASSERT_EQ(read_buffer_span.subspan(0, offset), zeros_span.subspan(0, offset));
    ASSERT_EQ(read_buffer_span.subspan(offset, length), buffer_span.subspan(0, length));
    ASSERT_EQ(read_buffer_span.subspan(offset + length), zeros_span.subspan(offset + length));

    // truncate first
    if (length > 0) {  // if length == 0, the file isn't extended to the offset
      ASSERT_STATUS_OK(WriteDataToFile({}, tmp.path));

      ASSERT_STATUS_OK(Env::Default().WriteBufferIntoFile(buffer_span, tmp.path.c_str(), offset, length));

      ASSERT_STATUS_OK(ReadDataFromFile(tmp.path, read_buffer_span.subspan(0, offset + length)));

      ASSERT_EQ(read_buffer_span.subspan(offset, length), buffer_span.subspan(0, length));
    }
  }

  // invalid - negative offset
  ASSERT_FALSE(Env::Default().WriteBufferIntoFile(buffer_span, tmp.path.c_str(), -1, buffer_span.size()).IsOK());

  // invalid - length too long
  ASSERT_FALSE(Env::Default().WriteBufferIntoFile(buffer_span, tmp.path.c_str(), 0, buffer_span.size() + 1).IsOK());

  // invalid - buffer too short
  ASSERT_FALSE(Env::Default().WriteBufferIntoFile(buffer_span.subspan(0, 2), tmp.path.c_str(), 0, 3).IsOK());
}

#ifndef _WIN32  // not implemented on Windows
TEST(FileIoTest, MapFileIntoMemory) {
  static const auto page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);

  TempFilePath tmp(ORT_TSTR("map_file_test_"));
  const auto expected_data = GenerateData(page_size * 3 / 2);
  ASSERT_STATUS_OK(WriteDataToFile(gsl::make_span(expected_data), tmp.path));

  const auto offsets_and_lengths = GenerateValidOffsetLengthPairs(0, expected_data.size(), page_size / 10);

  for (const auto& offset_and_length : offsets_and_lengths) {
    const auto offset = offset_and_length.first;
    const auto length = offset_and_length.second;

    SCOPED_TRACE(MakeString("offset: ", offset, ", length: ", length));

    Env::MappedMemoryPtr mapped_memory{};
    ASSERT_STATUS_OK(Env::Default().MapFileIntoMemory(tmp.path.c_str(), offset, length, mapped_memory));

    auto mapped_span = gsl::make_span(mapped_memory.get(), length);

    auto expected_data_span = gsl::make_span(expected_data.data() + offset, length);

    ASSERT_EQ(mapped_span, expected_data_span);
  }

  {
    Env::MappedMemoryPtr mapped_memory{};

    // invalid - negative offset
    ASSERT_FALSE(Env::Default().MapFileIntoMemory(tmp.path.c_str(), -1, 0, mapped_memory).IsOK());
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
