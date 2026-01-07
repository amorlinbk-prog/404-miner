import yaml
from pathlib import Path
from pydantic import BaseModel, Field


class VLMModelSettings(BaseModel):
    model_id: str = Field(..., description="the HF VLM repository to use.")
    max_tokens: int = Field(..., description="max tokens that will be used during generation.")
    max_model_len: int = Field(..., description="max tokens that will cover the input instruction length.")
    temperature: float = Field(..., description="temperature for the VLM model.")
    top_p: float = Field(..., description="top p tokens will be returned in VLM response.")
    seed: int = Field(..., description="random seed for the VLM")
    tensor_parallel_size: int = Field(..., description="the amount of GPUs to use.")
    disable_sliding_window: bool = Field(..., description="enable/disable sliding window for the VLM model.")
    enable_chunked_prefill: bool = Field(..., description="enable/disable chunked prefill")
    disable_mm_preprocessor_cache: bool = Field(..., description="enable/disable mm preprocessor cache")
    cpu_offload_gb: int = Field(..., description="default 0 GB, defines how many GB to offload to RAM.")
    model_precision: str = Field(..., description="default model precision, e.g. float16, bfloat16 etc.")
    max_num_batched_tokens: int = Field(..., description="the batch size for the batch prefill.")


class VLMModelParams(BaseModel):
    model_billions: float = Field(..., description="amount of parameters (billions) for current model.")
    hidden_size_llm: int = Field(..., description="the size of the hidden layers for llm part")
    hidden_size_vision: int = Field(..., description="the size of the hidden layers for visual part")
    num_hidden_layers_llm: int = Field(..., description="number of hidden layers for llm part.")
    num_hidden_layers_vision: int = Field(..., description="number of hidden layers for vision part.")
    num_attention_heads_llm: int = Field(..., description="number of attention layers for llm part.")
    num_attention_heads_vision: int = Field(..., description="number of attention layers for vision part.")
    num_key_value_heads: int = Field(..., description="number of kv heads.")
    intermediate_size_llm: int = Field(..., description="intermediate layers size llm part.")
    intermediate_size_vision: int = Field(..., description="intermediate layers size vision part.")
    image_size: int = Field(..., description="the size of the image that is accepted by the model.")
    patch_size: int = Field(..., description="the size of the patch.")


class VLMSettings(BaseModel):
    instruction_img_selection: str = Field(...,description="Instruction for the VLM for img to 3d pipeline.")
    vlm_model: VLMModelSettings
    vlm_model_params: VLMModelParams


def load_vlm_image_selector_settings_from_yaml(file_path: Path) -> VLMSettings:
    with file_path.open() as f:
        config_data = yaml.safe_load(f)
    return VLMSettings.model_validate(config_data)