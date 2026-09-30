#pragma once
#include "artifact/materializer.h"
#include "targets/falcon_h1/falcon_h1_binder.h"
#include "ops/linear/gptq/gptq_g128.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
namespace ninfer::targets::falcon_h1 {
class GptqExecution {
public:
    ~GptqExecution();
    void stage(const BoundModel&, artifact::MaterializedArtifact&);
    void linear(const std::string& projection, const float* x, float* y, std::int32_t m, cudaStream_t) const;
    void reset();
    [[nodiscard]] bool staged() const noexcept { return !views_.empty(); }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
private:
    std::map<std::string, ops::detail::GptqG128View> views_;
    artifact::MaterializedArtifact* owner_ = nullptr;
    std::weak_ptr<int> lifetime_;
    std::uint64_t generation_ = 0;
};
}
