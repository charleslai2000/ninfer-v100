#pragma once

#include "falcon_h1_binder.h"
#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/candidate_continuation.h"
#include <cuda_runtime_api.h>
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::targets::falcon_h1 {
namespace qwen_store = ninfer::targets::qwen3_6::detail;
struct ModelGeometry {
    std::uint64_t hidden_size=0, intermediate_size=0, vocab_size=0, layer_count=0;
    std::uint64_t attention_heads=0, kv_heads=0, head_dim=0, conv_width=0;
    std::uint64_t ssm_heads=0, ssm_state=0, ssm_chunk=0, ssm_groups=0, ssm_head_dim=0;
    std::uint64_t context_limit=0, recurrent_conv_channels=0, recurrent_conv_width=0;
    std::uint64_t recurrent_value_heads=0;
    std::uint64_t recurrent_value_head_dim=0, recurrent_key_head_dim=0;
    double embedding_multiplier=0, attention_in_multiplier=0, attention_out_multiplier=0;
    double key_multiplier=0, ssm_in_multiplier=0, ssm_out_multiplier=0;
    double rms_norm_eps=0, rope_theta=0;
    std::vector<double> ssm_multipliers, mlp_multipliers;
    std::vector<std::string> layer_types;
};
[[nodiscard]] ModelGeometry geometry_from_artifact(const BoundModel&, std::uint64_t context_limit);
enum class LayerStage : std::uint8_t { SharedPreNorm, AttentionBranch, MambaBranch, ResidualMerge, PreFfnNorm, GptqFfn };
struct LayerDispatchSlots { std::uint32_t layer=0; std::vector<LayerStage> stages; };
struct FalconContinuationSnapshot { ContinuationDescriptor descriptor; };
class Program {
public:
    using KVStore=qwen_store::KVAddressSpaceStore; using StateStore=qwen_store::StateImageStore;
    Program(ModelGeometry, std::uint32_t kv_entitlement, std::int32_t row_count,
            std::uint32_t sequence_capacity, KVStore&, StateStore&,
            qwen3_6::StateImageDevicePool&);
    ~Program();
    [[nodiscard]] const ModelGeometry& geometry() const noexcept { return geometry_; }
    [[nodiscard]] const std::vector<LayerDispatchSlots>& layers() const noexcept { return layers_; }
    [[nodiscard]] std::size_t sequence_count() const noexcept { return sequences_.size(); }
    [[nodiscard]] KVStore& kv_store() noexcept { return *kv_; }
    [[nodiscard]] const KVStore& kv_store() const noexcept { return *kv_; }
    [[nodiscard]] const KVPageGeometry& kv_geometry() const noexcept { return kv_->physical_geometry(); }
    [[nodiscard]] std::size_t attention_workspace_bytes(std::size_t tokens) const;
    [[nodiscard]] std::size_t configure_attention_kv_storage(LayoutBuilder& layout,
                                                              std::uint32_t physical_page_count) const;
    void run_attention(std::uint64_t sequence_id, std::uint32_t layer, const BoundModel&,
                       const artifact::MaterializedArtifact&, const Tensor& input, Tensor& output,
                       DeviceSpan workspace, std::size_t workspace_bytes, cudaStream_t stream);
    [[nodiscard]] std::uint64_t create_sequence(std::int32_t row);
    [[nodiscard]] ContinuationOwner& sequence(std::uint64_t id);
    [[nodiscard]] const ContinuationOwner& sequence(std::uint64_t id) const;
    void set_failure_injection_for_test(std::uint64_t id, std::int32_t stage);
    [[nodiscard]] FalconContinuationSnapshot fork_snapshot(std::uint64_t id) const;
    void reset_sequence(std::uint64_t id, std::int32_t row);
    void release_sequence(std::uint64_t id);
private:
    struct Sequence;
    ModelGeometry geometry_; std::uint32_t kv_entitlement_; std::int32_t row_count_;
    std::uint32_t sequence_capacity_; KVStore* kv_; StateStore* states_;
    qwen3_6::StateImageDevicePool* state_physical_;
    std::uint64_t next_id_=1; std::vector<LayerDispatchSlots> layers_;
    std::vector<std::unique_ptr<Sequence>> sequences_;
};
} // namespace ninfer::targets::falcon_h1
