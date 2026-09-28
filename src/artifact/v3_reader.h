#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace ninfer::artifact::v3 {
struct GptqG128Geometry { std::uint64_t k,n,groups,qweight_bytes,qzeros_bytes,scales_bytes,qzeros_offset,scales_offset,bytes; };
GptqG128Geometry gptq_g128_geometry(std::uint64_t k, std::uint64_t n);

// Formal, target-independent v3 reader. The first files entry is the entry
// file (path=null); later entries are continuation parts.
class Reader {
public:
 explicit Reader(const std::filesystem::path& entry);
 const nlohmann::json& directory() const noexcept { return directory_; }
 const std::array<std::byte,16>& artifact_id() const noexcept { return artifact_id_; }
 std::uint64_t payload_offset() const noexcept { return payload_offset_; }
 std::uint64_t logical_payload_bytes() const noexcept { return logical_payload_bytes_; }
 std::vector<std::byte> payload(const std::string& name) const;
 std::vector<std::byte> range(std::uint64_t offset, std::uint64_t bytes) const;
private:
 struct Part { std::filesystem::path path; std::uint64_t logical_offset{}, payload_bytes{}; };
 nlohmann::json directory_;
 std::array<std::byte,16> artifact_id_{};
 std::vector<Part> parts_;
 std::uint64_t payload_offset_ = 0;
 std::uint64_t logical_payload_bytes_ = 0;
};
}
