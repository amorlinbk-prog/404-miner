import torch
from loguru import logger
from pydantic import BaseModel
from background_remover.config import VLMModelParams


class VRAMInfo(BaseModel):
    allocated_memory_gb: float
    peak_memory_gb: float
    reserved_memory: float
    total_vram_gb: float
    free_vram_gb: float


class VRAMUsageEstimator:
    def __init__(self, gpu_id: int = 0) -> None:
        self._gpu_id = gpu_id
        self._bytes_per_param = {
            "float32": {
                "parameter_data_type_size": 4,
                "kv_data_type_size": 4
            },
            "float16": {
                "parameter_data_type_size": 2,
                "kv_data_type_size": 2
            },
            "bfloat16": {
                "parameter_data_type_size": 2,
                "kv_data_type_size": 2
            },
            "int8":{
                "parameter_data_type_size": 1,
                "kv_data_type_size": 1
            },
            "int4":{
                "parameter_data_type_size": 0.5,
                "kv_data_type_size": 0.5
            }
        }
        self._model_id: str = ""

    def get_current_gpu_vram_usage(self) -> VRAMInfo:
        """ Function for getting information about GPU VRAM.  """

        if torch.cuda.is_available():
            logger.info(f"CUDA device found: {torch.cuda.get_device_name(self._gpu_id)}")
            current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

            return VRAMInfo(
                allocated_memory_gb=round(current_memory, 2),
                peak_memory_gb=round(max_memory, 2),
                total_vram_gb=round(total_memory, 2),
                reserved_memory=round(reserved_memory, 2),
                free_vram_gb=round(total_memory-current_memory, 2)
            )
        else:
            logger.warning("CUDA device was not found!")
            return VRAMInfo(
                allocated_memory_gb=0,
                peak_memory_gb=0,
                total_vram_gb=0,
                reserved_memory=0,
                free_vram_gb=0
            )

    def _estimate_base_model_vram_consumption(self, precision: str, model_billions: float) -> float:
        """ Function for estimating base model VRAM consumption. """

        if precision not in self._bytes_per_param:
            raise ValueError(f"Unsupported precision: {precision}")

        number_of_model_parameters = model_billions * 1e9
        model_weight_bytes = number_of_model_parameters * self._bytes_per_param[precision]["parameter_data_type_size"]
        model_weight_gb = model_weight_bytes / (1000 ** 3)
        return  model_weight_gb

    def _estimate_vram_consumption_for_llm_part(
            self,
            input_tokens_number: int,
            output_tokens_number: int,
            precision: str,
            max_num_sequences: int,
            model_parameters_settings: VLMModelParams
    ) -> float:
        """ Function for estimating VRAM consumption for LLM part of the base model. """

        # Component 1: Overhead Memory (Static)
        # Fixed memory for non-PyTorch components like CUDA kernels, etc.
        overhead_memory_gb = 3.0

        # Component 2: PyTorch Activation Peak Memory (Dynamic)
        # Memory for storing intermediate calculations (activations) during the forward pass.
        # This scales with max_num_sequences and sequence length.
        sequence_length = input_tokens_number + output_tokens_number
        pytorch_activation_peak_memory_bytes = (
                max_num_sequences *
                sequence_length * (
                        18 * model_parameters_settings.hidden_size_llm +
                        4 * model_parameters_settings.intermediate_size_llm
                )
        )
        pytorch_activation_peak_memory_gb = pytorch_activation_peak_memory_bytes / (1000 ** 3)

        # Component 3: KV Cache Memory (Dynamic)
        # Memory for the Key-Value cache, which stores attention context to speed up token generation.
        # This scales with max_num_sequences and sequence length.
        kv_vectors = 2  # One for Key, one for Value
        head_dims = model_parameters_settings.hidden_size_llm // model_parameters_settings.num_attention_heads_llm

        kv_cache_memory_per_batch_bytes = (
                kv_vectors *
                max_num_sequences *
                sequence_length *
                model_parameters_settings.num_key_value_heads *
                head_dims *
                model_parameters_settings.num_hidden_layers_llm *
                self._bytes_per_param[precision]["kv_data_type_size"]
        )
        kv_cache_memory_per_batch_gb = kv_cache_memory_per_batch_bytes / (1000 ** 3)

        # --- Final Calculation ---
        # Sum of static and dynamic memory components.
        required_gpu_memory_gb = (
                overhead_memory_gb +
                pytorch_activation_peak_memory_gb +
                kv_cache_memory_per_batch_gb
        )

        return required_gpu_memory_gb

    def _estimate_vram_consumption_for_vision_part(
            self,
            image_number: int,
            image_shape: tuple[int, int, int],
            input_tokens_number: int,
            precision: str,
            max_num_sequences: int,
            model_parameters_settings: VLMModelParams
    ) -> float:
        """ Function for estimating VRAM consumption for Vision part of the base model. """

        # 0. image patches info precomputation
        patches_per_image = ((image_shape[0] // model_parameters_settings.patch_size) *
                             (image_shape[1] // model_parameters_settings.patch_size))
        hidden_size = model_parameters_settings.hidden_size_vision

        # 1. Raw image tensor memory (input images)
        raw_image_memory_bytes = (
                max_num_sequences *
                image_number *
                image_shape[0] *
                image_shape[1] *
                image_shape[2] *
                self._bytes_per_param[precision]["parameter_data_type_size"]
        )
        raw_image_memory_gb = raw_image_memory_bytes / (1000 ** 3)

        # 2. Image preprocessing memory (resized/normalized images)
        preprocessing_memory_gb = raw_image_memory_gb * 0.5  # Conservative estimate for preprocessing buffers

        # 3. Vision features memory (image patches -> embeddings)
        vision_features_memory_bytes = (
                max_num_sequences *
                image_number *
                patches_per_image *
                hidden_size *
                self._bytes_per_param[precision]["parameter_data_type_size"]
        )
        vision_features_memory_gb = vision_features_memory_bytes / (1000 ** 3)

        # 4. Cross-modal attention memory (vision-text interaction)
        cross_attention_memory_bytes = (
                max_num_sequences *
                image_number *
                patches_per_image *
                input_tokens_number *
                self._bytes_per_param[precision]["parameter_data_type_size"]
        )
        cross_attention_memory_gb = cross_attention_memory_bytes / (1000 ** 3)

        # Component 5: PyTorch Activation Peak Memory (Dynamic)
        total_tokens = input_tokens_number + (image_number * patches_per_image)

        # estimate per-token activation footprint (empirical constant)
        activation_bytes_per_token = 6 * model_parameters_settings.hidden_size_vision  # empirical ~5–8×hidden_size

        pytorch_activation_peak_memory_bytes = max_num_sequences * total_tokens * activation_bytes_per_token
        pytorch_activation_peak_memory_gb = pytorch_activation_peak_memory_bytes / (1000 ** 3)

        # --- Final Calculation ---
        # Sum of static and dynamic memory components.
        required_gpu_memory_gb = (
                pytorch_activation_peak_memory_gb +
                raw_image_memory_gb +
                preprocessing_memory_gb +
                vision_features_memory_gb +
                cross_attention_memory_gb
        )

        return required_gpu_memory_gb

    def estimate_vram_for_llm(
            self,
            input_tokens_number: int,
            output_tokens_number: int,
            precision: str,
            max_num_sequences: int,
            model_parameters_settings: VLMModelParams
    ) -> float:
        """ Function for estimating VRAM consumption for the LLM model. """

        base_model_vram_gb = self._estimate_base_model_vram_consumption(
            precision, model_parameters_settings.model_billions
        )

        llm_part_vram_gb = self._estimate_vram_consumption_for_llm_part(
            input_tokens_number=input_tokens_number,
            output_tokens_number=output_tokens_number,
            precision=precision,
            max_num_sequences=max_num_sequences,
            model_parameters_settings=model_parameters_settings
        )

        return base_model_vram_gb + llm_part_vram_gb

    def estimate_vram_for_vlm(
            self,
            image_number: int,
            image_shape: tuple[int, int, int],
            input_tokens_number: int,
            output_tokens_number: int,
            precision: str,
            max_num_sequences: int,
            model_parameters_settings: VLMModelParams
    ) -> float:
        """ Function for estimating VRAM consumption for the VLM model. """

        llm_part_vram = self.estimate_vram_for_llm(
            input_tokens_number=input_tokens_number,
            output_tokens_number=output_tokens_number,
            precision=precision,
            max_num_sequences=max_num_sequences,
            model_parameters_settings=model_parameters_settings
        )

        vision_part_vram_gb = self._estimate_vram_consumption_for_vision_part(
            image_number=image_number,
            image_shape=image_shape,
            input_tokens_number=input_tokens_number,
            precision=precision,
            max_num_sequences=max_num_sequences,
            model_parameters_settings=model_parameters_settings
        )

        return llm_part_vram + vision_part_vram_gb

    def get_gpu_mem_utilization_coeff(self, estimated_vram: float) -> float:
        """ Function for estimating gpu mem utilization coefficient. """

        vram_info = self.get_current_gpu_vram_usage()
        free_vram = max(vram_info.free_vram_gb, 1e-6)
        ratio = estimated_vram / free_vram

        mem_coeff = max(ratio, 0.15)
        mem_coeff = min(mem_coeff, 0.9)
        return mem_coeff
