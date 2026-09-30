"""MXFP4 W4A16 MoE experts with the SiTU activation (Kimi K3) for CPU."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.platforms import current_platform

from . import ops


class K3CpuExpertsMxfp4Situ(mk.FusedMoEExpertsModular):
    """Weights stay in checkpoint layout: w13 [E, 2I, K/2] (gate rows, then up
    rows), w2 [E, K, I/2], E8M0 scales per 32 elements."""

    @staticmethod
    def convert_weights(
        *,
        layer,
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w13_bias=None,
        w2_bias=None,
        **_,
    ):
        assert w13_bias is None and w2_bias is None, "Kimi K3 experts have no bias"
        return w13_weight, w2_weight, w13_weight_scale, w2_weight_scale, None, None

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SITU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return True

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None, routing_method: RoutingMethodType
    ) -> bool:
        return True

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ) -> None:
        assert not apply_router_weight_on_input
        cfg = self.moe_config
        output.copy_(
            ops.moe_mxfp4_situ(
                hidden_states,
                w1,
                self.w1_scale,
                w2,
                self.w2_scale,
                topk_weights,
                topk_ids,
                cfg.activation_situ_beta,
                cfg.activation_situ_linear_beta,
                expert_map,
            )
        )
