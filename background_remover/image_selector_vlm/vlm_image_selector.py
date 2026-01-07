import gc
import re
from PIL import Image


import torch
from transformers import AutoTokenizer
from loguru import logger
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from background_remover.config import VLMSettings
from background_remover.image_selector_vlm.vram_memory_estimator import VRAMUsageEstimator
from background_remover.utils.rand_utils import secure_randint


class VLMImageSelector:
    def __init__(
            self,
            vlm_settings: VLMSettings,
            image_number_in_use: int,
            image_shape: tuple[int, int, int],
            kv_cache_dtype: str="fp8_e4m3"
    ) -> None:
        self._model: LLM | None = None
        self.settings: VLMSettings = vlm_settings
        self._seed = -1

        self._sampling_parameters: SamplingParams | None = None
        self._stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        self._stop_token_ids = []
        self._tokenizer = AutoTokenizer.from_pretrained(self.settings.vlm_model.model_id)

        self._vram_usage_estimator = VRAMUsageEstimator()
        self._image_number_in_use = image_number_in_use
        self._image_shape = image_shape
        self._kv_cache_dtype = kv_cache_dtype

        self._guided_decoding_params_json = GuidedDecodingParams(choice=["left", "right"], backend="xgrammar")

    def load_model(self, custom_mem_util: float | None=None) -> None:
        """ Function for preloading LLM model in GPU memory """

        logger.info(f"Used VLM model for image selection is: {self.settings.vlm_model}")
        vram_info = self._vram_usage_estimator.get_current_gpu_vram_usage()

        if custom_mem_util is None:
            estimated_vlm_memory_gb = self._vram_usage_estimator.estimate_vram_for_vlm(
                image_number=self._image_number_in_use,
                image_shape=self._image_shape,
                input_tokens_number=self.settings.vlm_model.max_model_len,
                output_tokens_number=self.settings.vlm_model.max_tokens,
                precision=self.settings.vlm_model.model_precision,
                max_num_sequences=1,
                model_parameters_settings=self.settings.vlm_model_params
            )
            mem_util = self._vram_usage_estimator.get_gpu_mem_utilization_coeff(estimated_vlm_memory_gb)

            logger.info(f"Available VRAM memory: {vram_info.free_vram_gb} GB")
            logger.info(f"Estimated needed VRAM memory: {estimated_vlm_memory_gb} GB")
            logger.info(f"VRAM memory utilization coeff.: {mem_util}")

        else:
            mem_util = custom_mem_util

            logger.info(f"GPU Info [GB]: {vram_info}")
            logger.info(f"Available VRAM memory: {vram_info.free_vram_gb} GB")
            logger.info(f"VRAM memory utilization coeff.: {mem_util}")

        if mem_util > 1.0:
            raise RuntimeError(f"Not enough available VRAM for running VLM model!")

        self._model = LLM(
            model=self.settings.vlm_model.model_id,
            dtype=self.settings.vlm_model.model_precision,
            trust_remote_code=True,
            tensor_parallel_size=self.settings.vlm_model.tensor_parallel_size,
            gpu_memory_utilization=mem_util,
            max_model_len=self.settings.vlm_model.max_model_len,
            disable_mm_preprocessor_cache=self.settings.vlm_model.disable_mm_preprocessor_cache,
            enable_chunked_prefill=self.settings.vlm_model.enable_chunked_prefill,
            max_num_batched_tokens=self.settings.vlm_model.max_num_batched_tokens,
            cpu_offload_gb = self.settings.vlm_model.cpu_offload_gb,
            kv_cache_dtype=self._kv_cache_dtype,
            calculate_kv_scales=True,
            max_num_seqs=1,
            limit_mm_per_prompt={"image": self._image_number_in_use, "video": 0, "audio": 0}
        )

        self._stop_token_ids = [self._tokenizer.convert_tokens_to_ids(i) for i in self._stop_tokens]

    def unload_model(self) -> None:
        """ Function for unloading the model """
        logger.info("Unloading model from GPU VRAM.")

        destroy_model_parallel()
        del self._model.llm_engine
        del self._model
        gc.collect()
        torch.cuda.empty_cache()
        self._model = None

    def create_sampling_params(self, seed: int) -> None:
        """ Function for creating sampling parameters for the vLLM backend """

        if seed < 0:
            seed = secure_randint(0, 10000)
        else:
            seed = seed

        sampling_params = SamplingParams(
            n=1,
            temperature=self.settings.vlm_model.temperature,
            max_tokens=self.settings.vlm_model.max_tokens,
            top_p=self.settings.vlm_model.top_p,
            stop_token_ids=self._stop_token_ids,
            seed=seed,
            guided_decoding=self._guided_decoding_params_json
        )
        self._sampling_parameters = sampling_params

    def create_chat_template(self, images_num: int, instruction_prompt: str) -> str:
        """ Function for creating chat template for the model that will be used. """
        placeholders = "\n".join(f"Image-{i}: <image>\n" for i in range(1, images_num+1))
        messages = [{
            'role': 'user',
            'content': f"{placeholders}\n{instruction_prompt}"
        }]
        prompt_template = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt_template

    def create_input_template(self, images: list[Image.Image], instruction_prompt: str) -> dict:
        """ Function for creating final template that will be using with LLM.generate() function. """
        prompt_template = self.create_chat_template(len(images), instruction_prompt)
        input_template = {
            "prompt": prompt_template,
            "multi_modal_data": {
                "image": images
            }
        }

        return input_template

    def select(self, images: list[Image.Image]) -> int:
        """ Function that calls vLLM API for generating prompts. """
        logger.info("Processing input query ... ")

        instruction = self.settings.instruction_img_selection
        input_template = self.create_input_template(images, instruction)

        output = self._model.generate(input_template, sampling_params=self._sampling_parameters)
        reply = output[0].outputs[0].text

        logger.warning(f"reply: {reply}")
        selected_image_ind = self.answer_parser(reply)

        return selected_image_ind

    def _find_whole_word(self, search_string, input_string):
        """ Helping parser function of the answer provided by the model. """
        result = re.compile(r'\b({0})\b'.format(search_string), flags=re.IGNORECASE).search(input_string)
        return True if result is not None else False

    def answer_parser(self, answer: str) -> int:
        """Functon for parsing the answer of the model"""
        input_prompt = answer.lower()

        if (self._find_whole_word("left", input_prompt) or
            self._find_whole_word("image-1", input_prompt) or
            self._find_whole_word("image 1", input_prompt)):
            logger.warning("Selected image LEFT")
            return 0
        elif (self._find_whole_word("right", input_prompt) or
              self._find_whole_word("image-2", input_prompt) or
              self._find_whole_word("image 2", input_prompt)):
            logger.warning("Selected image RIGHT")
            return 1
        else:
            logger.warning("Model did not answer the question! Return 1 by default.")
            return 1