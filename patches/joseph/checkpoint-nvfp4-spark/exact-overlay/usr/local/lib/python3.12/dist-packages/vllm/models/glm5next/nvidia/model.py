# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): the model classes declare SupportsReplaySSM; the DFlash2 auxiliary
# hidden-state capture (EagleModelMixin, SupportsEagle3) adapted from vLLM's DeepseekV4Model.forward.

from collections.abc import Iterable
import sys
from typing import ClassVar, Literal

import torch
from torch import nn

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
import os as _glm53_os

_GLM53_ROUTER_DEDUP = _glm53_os.environ.get("VLLM_GLM53_ROUTER_DEDUP", "0").strip() == "1"

from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mhc import (
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
    hc_contract,
    hc_expand,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
    SupportsReplaySSM,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from .attention import Glm5NextMLAAttention
from .kda import Glm5NextLinearAttention
from .multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
    Glm5NextVisionTransformer,
)

logger = init_logger(__name__)


_GLM53_MXFP8_PROJECTIONS = {
    # Indexer: fuse wk and weights_proj; wq_b is direct.
    ".indexer.wk.": (
        ".indexer.wk_weights_proj",
        0,
        "indexer_wk",
        "FUSED",
        True,
    ),
    ".indexer.weights_proj.": (
        ".indexer.wk_weights_proj",
        1,
        "indexer_weights_proj",
        "FUSED",
        False,
    ),
    ".indexer.wq_b.": (
        ".indexer.wq_b",
        None,
        "indexer_wq_b",
        "DIRECT",
        False,
    ),
    # KDA: merge q, k, v, b, f_a, g_a into one GEM; f_b/g_b/o are direct.
    ".q_proj.": (".in_proj_qkvbfg_a", 0, "q_proj", "FUSED", False),
    ".k_proj.": (".in_proj_qkvbfg_a", 1, "k_proj", "FUSED", False),
    ".v_proj.": (".in_proj_qkvbfg_a", 2, "v_proj", "FUSED", False),
    ".b_proj.": (".in_proj_qkvbfg_a", 3, "b_proj", "FUSED", False),
    ".f_a_proj.": (".in_proj_qkvbfg_a", 4, "f_a_proj", "FUSED", False),
    ".g_a_proj.": (".in_proj_qkvbfg_a", 5, "g_a_proj", "FUSED", False),
    ".f_b_proj.": (".f_b_proj", None, "f_b_proj", "DIRECT", False),
    ".g_b_proj.": (".g_b_proj", None, "g_b_proj", "DIRECT", False),
    # MLA: fuse q_a/kv_a; q_b/kv_b/o are direct BF16 targets.
    ".q_a_proj.": (
        ".fused_qkv_a_proj",
        0,
        "mla_q_a_proj",
        "FUSED",
        True,
    ),
    ".kv_a_proj_with_mqa.": (
        ".fused_qkv_a_proj",
        1,
        "mla_kv_a_proj_with_mqa",
        "FUSED",
        True,
    ),
    ".q_b_proj.": (".q_b_proj", None, "mla_q_b_proj", "DIRECT", True),
    ".kv_b_proj.": (".kv_b_proj", None, "mla_kv_b_proj", "DIRECT", True),
    ".o_proj.": (".o_proj", None, "o_proj", "DIRECT", True),
}
_GLM53_MXFP8_MARKED = set()
_GLM53_CONTRACT_TRACE_ENABLED = (
    _glm53_os.environ.get("VLLM_GLM53_CONTRACT_TRACE", "0").strip() == "1"
)
_GLM53_CONTRACT_TRACE_COUNT = 0


def _glm53_contract_trace(event: str, **fields) -> None:
    """Emit a bounded load-time name trace without changing loader behavior."""
    global _GLM53_CONTRACT_TRACE_COUNT
    if not _GLM53_CONTRACT_TRACE_ENABLED or _GLM53_CONTRACT_TRACE_COUNT >= 128:
        return
    _GLM53_CONTRACT_TRACE_COUNT += 1
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"GLM53_CONTRACT_TRACE event={event} {details}",
        file=sys.stderr,
        flush=True,
     )


def _try_load_glm53_mxfp8_projection(
    name,
    tensor,
    buf,
    params_dict,
    loaded_params,
    kv_a_pad_size,
) -> bool:
    """Load B MXFP8 constituents into existing BF16 direct/fused targets.

    The NVFP4-Spark checkpoint provides MXFP8 E4M3 ``weight`` plus U8 E8M0
    ``weight_scale`` pairs.  RiNGSiDE's pinned KDA, Indexer, and MLA targets
    are BF16 (including the fused targets), so decode each constituent only
    when the existing runtime target has no native scale parameter.  This is a
    checkpoint-contract mapping only: it preserves source constituent
    boundaries, uses the existing target weight loader, and never creates a
    synthetic fused scale.
    """
    matched = None
    for suffix, target in _GLM53_MXFP8_PROJECTIONS.items():
        if suffix in name:
            matched = (suffix, target)
            break
    if matched is None:
        return False

    suffix, (target_base, shard_id, label, mapping_mode, legacy_overlap) = matched
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = name.endswith(".weight_scale") and tensor.dtype == torch.uint8
    _glm53_contract_trace(
        "ADAPTER_MATCH",
        source=name,
        family=label,
        dtype=tensor.dtype,
        is_weight=is_weight,
        is_scale=is_scale,
    )
    if not is_weight and not is_scale:
        _glm53_contract_trace(
            "ADAPTER_RETURN",
            source=name,
            family=label,
            handled=False,
            terminal_action="NOT_A_B_MXFP8_TENSOR",
        )
        return False

    layer_prefix = name.rsplit(suffix, 1)[0]
    target_w = f"{layer_prefix}{target_base}.weight"
    target_s = f"{layer_prefix}{target_base}.weight_scale"
    if target_w not in params_dict or target_s in params_dict:
        # A quantized target owns its native scale; leave it to the normal path.
        _glm53_contract_trace(
            "ADAPTER_RETURN",
            source=name,
            family=label,
            handled=False,
            terminal_action="QUANTIZED_TARGET_NATIVE_PATH",
        )
        return False

    if legacy_overlap and a "¶»§q«^u¶¬{®¢{^žÚ&ŠÛ^uú+n·¯ŠÜ¢žØb²Ë¦™ªò