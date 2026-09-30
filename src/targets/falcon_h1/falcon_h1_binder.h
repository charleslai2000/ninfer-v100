#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "nlohmann/json.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ninfer::targets::falcon_h1 {

struct BoundTensor {
    artifact::ObjectHandle handle;
    std::string role;
};

struct GptqPermutationStage { artifact::ObjectHandle canonical; std::size_t offset = 0; std::size_t bytes = 0; };
struct BoundModel {
    nlohmann::json config;
    std::map<std::string, GptqPermutationStage> gptq_permutations;
    std::map<std::string, std::vector<std::int32_t>> gptq_host_permutations;
    std::map<std::string, std::array<std::uint64_t, 3>> gptq_geometry;
    std::map<std::string, BoundTensor> tensors;
    artifact::MaterializationPlan materialization;
    std::map<std::string, std::uint64_t> device_bytes_by_format;
    std::map<std::string, double> config_scalars;
    std::map<std::string, std::vector<double>> config_arrays;
    std::map<std::string, std::uint64_t> config_integers;
    std::vector<std::string> layer_types;
    std::size_t layer_count = 0;
    std::size_t resource_count = 0;
};

// Validates the immutable v3 target contract and creates the exact device placement
// plan. This is a loader/binder only; it does not execute model operations.
BoundModel bind(artifact::Reader& reader);
BoundModel bind_config_only(const nlohmann::json& config);
void validate(const artifact::Reader& reader);
void validate_directory(const nlohmann::json& directory);

} // namespace ninfer::targets::falcon_h1
