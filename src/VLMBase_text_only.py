from abc import ABC, abstractmethod
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import sys
import gc
import argparse
import re
from transformers import pipeline
import time
import gc
from PIL import Image
from pathlib import Path
from transformers import FineGrainedFP8Config

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)

# Gemma 3 (si está disponible en tu transformers)
try:
    from transformers import Gemma3ForConditionalGeneration
    HAS_GEMMA3 = True
except Exception:
    HAS_GEMMA3 = False

try:
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    HAS_MISTRAL3 = True
except Exception:
    Mistral3ForConditionalGeneration = None
    MistralCommonBackend = None
    HAS_MISTRAL3 = False
    
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bnb_config_4bits = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

bnb_config_8bits = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)


def read_txt_file_as_string(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content


def clean_gpu_cache():
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")
    gc.collect()
    print("Garbage collection run.")
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()


# -------------------------------------------------------------------
#   PREPROCESADO DE IMAGEN PARA INTERNVL (adaptado de tu snippet)
# -------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_internvl(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# -------------------------------------------------------------------
#                        CLASE VLMModel
# -------------------------------------------------------------------

class VLMModel:
    """
    Wrapper genérico para modelos Vision-Language:

    - Gemma3 VLM  (Gemma3ForConditionalGeneration + AutoProcessor)
    - LLaVA 1.5   (LlavaForConditionalGeneration + AutoProcessor)
    - LLaVA 1.6   (LlavaNextForConditionalGeneration + LlavaNextProcessor)
    - Qwen3-VL    (Qwen3VLForConditionalGeneration + apply_chat_template)
    - Qwen3.5-35B-A3B (Qwen3VLForConditionalGeneration + apply_chat_template)
    - InternVL3.5 (AutoModel con trust_remote_code=True + tokenizer + model.chat)
    - Ministral 3 (Mistral3ForConditionalGeneration + MistralCommonBackend)
    """

    _model_cache = {}

    def __init__(self, model_name, quantization_config, temperature=0.7, top_k=50, top_p=0.9, do_sample=False):
        # nombres cortos -> rutas locales
        vlm_id_map = {
                    # --- Gemma  ---
                    "gemma3-4b-it": "google/gemma-3-4b-it",
                    "gemma3-12b-it": "google/gemma-3-12b-it",
                    "gemma3-27b-it": "google/gemma-3-27b-it",
                    "gemma4-31b-it": "google/gemma-4-31B-it",
        
                    # --- LLaVA (llava-hf) ---
                    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
                    "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
                    "llava-v1.6-mistral-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
                    "llava-v1.6-vicuna-13b": "liuhaotian/llava-v1.6-vicuna-13b",
        
                    # --- Qwen3-VL ---
                    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
                    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
                    "Qwen3-VL-32B-Instruct": "Qwen/Qwen3-VL-32B-Instruct",
                    "Qwen3.5-35B-A3B": "Qwen/Qwen3.5-35B-A3B",
                    "Qwen3-VL-32B-Thinking": "Qwen/Qwen3-VL-32B-Thinking-FP8",
                    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
        
                    # --- InternVL / OpenGVLab ---
                    "InternVL3_5-8B": "OpenGVLab/InternVL3_5-8B-Instruct",
                    "InternVL3_5-38B": "InternVL3_5-38B",
                    
                    # --- Mistral / Ministral 3 ---
                    "Ministral-3-14B": "mistralai/Ministral-3-14B-Instruct-2512"
        }

        if model_name not in vlm_id_map:
            raise ValueError(f"Modelo VLM no soportado: {model_name}")

        self.model_name = model_name
        self.model_id = vlm_id_map[model_name]

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

        if quantization_config == "4bits":
            self.quantization_config = bnb_config_4bits
        elif quantization_config == "8bits":
            self.quantization_config = bnb_config_8bits
        else:
            self.quantization_config = None

        self.device = device
        self.prompt = ""
        self.image = None      # PIL image (para Gemma / LLaVA / Qwen3)
        self.image_path = None
        self.response = ""
        self.text_only = False

        # flags para flujos especiales
        self.is_qwen3_chat = False
        self.is_qwen35_text_chat = False
        self.is_internvl = False
        self.is_gemma_chat = False
        self.is_llava_chat = False  # tanto 1.5 como 1.6 usan chat template
        self.is_ministral3 = False

        # objetos extra
        self.messages = None             # Gemma / Llava / Qwen3
        self.internvl_tokenizer = None   # tokenizer de InternVL
        self.pixel_values = None         # imagen preprocesada InternVL
        self.ministral_tokenizer = None

        cache_key = (self.model_name, quantization_config)
        if cache_key in VLMModel._model_cache:
            (self.model, self.processor, self.internvl_tokenizer,
             self.is_qwen3_chat, self.is_qwen35_text_chat, self.is_internvl,
             self.is_gemma_chat, self.is_llava_chat, self.is_ministral3) = VLMModel._model_cache[cache_key]
        else:
            self._load_model_and_processor()
            VLMModel._model_cache[cache_key] = (
                self.model,
                getattr(self, "processor", None),
                self.internvl_tokenizer,
                self.is_qwen3_chat,
                self.is_qwen35_text_chat,
                self.is_internvl,
                self.is_gemma_chat,
                self.is_llava_chat,
                self.is_ministral3,
            )

    # ---------------------------------------------------------
    #  CARGA AUTOMÁTICA DEL MODELO SEGÚN LA FAMILIA
    # ---------------------------------------------------------

    def _load_model_and_processor(self):

        base_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # ---------------- GEMMA 3 ----------------
        if self.model_name.startswith("gemma3"):
            if not HAS_GEMMA3:
                raise ImportError(
                    "Gemma3ForConditionalGeneration no está disponible en tu versión de transformers."
                )

            model_kwargs = {
                "device_map": "auto",
            }
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.bfloat16

            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id, **model_kwargs
            ).eval()

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.is_gemma_chat = True
        
        elif self.model_name.startswith("gemma4"):
            if not HAS_GEMMA3:
                raise ImportError(
                    "Gemma3ForConditionalGeneration no está disponible en tu versión de transformers."
                )

            model_kwargs = {
                "device_map": "auto",
                "attn_implementation":"sdpa",
            }
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.bfloat16

            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **model_kwargs
            ).eval()

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.is_gemma_chat = True


        # ---------------- LLaVA v1.6 (LlavaNext*) ----------------
        elif self.model_name.startswith("llava-v1.6"):
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": base_dtype,
                "low_cpu_mem_usage": True,
            }
            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config

            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id, **model_kwargs
            )
            self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
            self.is_llava_chat = True

        # ---------------- LLaVA v1.5 ----------------
        elif self.model_name.startswith("llava-1.5"):
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": base_dtype,
                "low_cpu_mem_usage": True,
            }
            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config

            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id, **model_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.is_llava_chat = True

        # ---------------- QWEN3-VL ----------------
        elif self.model_name.startswith("Qwen3-VL"):
            model_kwargs = {
                "dtype": "auto",
                "device_map": "auto",
            }
            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, **model_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.is_qwen3_chat = True

        # ---------------- Qwen3.5-27B (texto) ----------------
        elif self.model_name.startswith("Qwen3.5-27B"):
            model_kwargs = {
                "dtype": "auto",
                "device_map": "auto",
            }
            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config

            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.model_id, **model_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.is_qwen35_text_chat = True
            
        # ---------------- QWEN3.5 (texto) ----------------
        elif self.model_name.startswith("Qwen3.5"):
            model_kwargs = {
                "dtype": "auto",
                "device_map": "auto",
            }
            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config

            self.model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
                self.model_id, **model_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.is_qwen35_text_chat = True
            

        # ---------------- INTERNVL 3.5 ----------------
        elif self.model_name.startswith("InternVL3_5"):
            model_kwargs = {
                "dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }

            # only enable if installed
            try:
                import flash_attn  # noqa: F401
                model_kwargs["use_flash_attn"] = True
            except Exception:
                model_kwargs["use_flash_attn"] = False

            self.model = AutoModel.from_pretrained(
                self.model_id,
                **model_kwargs
            ).eval().cuda()

            self.internvl_tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=False,
            )
            self.processor = None
            self.is_internvl = True

        # ---------------- MINISTRAL 3 ----------------
        elif self.model_name.startswith("Ministral-3-14B"):
            self.ministral_tokenizer = MistralCommonBackend.from_pretrained(
                self.model_id
            )

            model_kwargs = {
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "quantization_config": FineGrainedFP8Config(
                    dequantize=True
                ),
            }

            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            ).eval()

            self.processor = self.ministral_tokenizer
            self.is_ministral3 = True
            

        else:
            raise ValueError(f"No sé qué loader usar para {self.model_name}")

    # ---------------------------------------------------------
    #  INTERFAZ
    # ---------------------------------------------------------

    def set_prompt(self, prompt, image_path=None, pil_image=None, text_only=False):
        """
        - Gemma / LLaVA / Qwen3 → usan PIL + AutoProcessor.
        - Gemma / LLaVA / Qwen3: usan messages + apply_chat_template.
        - InternVL: usa pixel_values (tiles) + model.chat.
        """

        self.prompt = prompt
        self.image_path = image_path
        self.image = None
        self.pixel_values = None
        self.text_only = bool(text_only)

        if self.text_only:
            self.messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            self.image = None
            self.image_path = None
            return
        # Qwen3.5-35B-A3B es de texto: no requiere imagen.
        if self.is_qwen35_text_chat:
            image = None
        else:
            # Para todos excepto InternVL: necesitamos una imagen PIL.
            if pil_image is not None:
                image = pil_image.convert("RGB")
            elif image_path is not None:
                image = Image.open(image_path).convert("RGB")
            else:
                if not self.is_internvl:
                    raise ValueError("Debes pasar una imagen (image_path o pil_image).")
                image = None

        self.image = image

        # ---------------- GEMMA3 (chat template) ----------------
        if self.is_gemma_chat:
            self.messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image},
                        {"type": "text", "text": self.prompt},
                    ],
                },
            ]

        # ---------------- LLaVA (1.5 y 1.6, chat template) ----------------
        elif self.is_llava_chat:
            # Igual que los ejemplos oficiales: texto + placeholder de imagen
            self.messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image"},
                    ],
                },
            ]

        # ---------------- QWEN3-VL (chat con template) ----------------
        elif self.is_qwen3_chat:
            self.messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]

        # ---------------- QWEN3.5 (chat) ----------------
        elif self.is_qwen35_text_chat:
            self.messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image},
                        {"type": "text", "text": self.prompt},
                    ],
                },
            ]
            
        # ---------------- MINISTRAL 3 ----------------
        elif self.is_ministral3:
            if self.image_path is None:
                raise ValueError(
                    "Ministral 3 requiere image_path; no puede usar solo pil_image."
                )

            image_path_abs = str(Path(self.image_path).resolve())

            self.messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"file://{image_path_abs}",
                            },
                        },
                    ],
                }
            ]

        # ---------------- INTERNVL (preprocesado tiles) ----------------
        if self.is_internvl:
            if self.image_path is None:
                raise ValueError("InternVL requiere image_path (ruta a archivo).")
            pixel_values = load_image_internvl(self.image_path, max_num=12)
            # dev = next(self.model.parameters()).device
            # self.pixel_values = pixel_values.to(dev, dtype=torch.bfloat16)
            if hasattr(self.model, "device") and self.model.device is not None:
                dev = self.model.device
            else:
                dev = next(self.model.parameters()).device

            self.pixel_values = pixel_values.to(dev, dtype=torch.bfloat16)

    # ---------------- Qwen3-VL: inferencia específica ----------------

    def _run_qwen3_vl_inference(self, max_new_tokens: int):
        if self.messages is None:
            raise RuntimeError("No hay messages definidos para Qwen3-VL. Llama antes a set_prompt().")

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=float(self.temperature),
            top_k=int(self.top_k) if self.top_k else None,
            top_p=float(self.top_p),
            do_sample=self.do_sample,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        self.response = output_text

    # ---------------- Qwen3.5-35B-A3B: inferencia específica----------------

    def _run_qwen35_inference(self, max_new_tokens: int):
        if self.messages is None:
            raise RuntimeError("No hay messages definidos para Qwen3.5. Llama antes a set_prompt().")

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=float(self.temperature),
                top_k=int(self.top_k) if self.top_k else None,
                top_p=float(self.top_p),
                do_sample=self.do_sample,
                # chat_template_kwargs={"enable_thinking": False},
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]

        self.response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # ---------------- Gemma3: inferencia específica ----------------

    def _run_gemma3_inference(self, max_new_tokens: int):
        if self.messages is None:
            raise RuntimeError("No hay messages definidos para Gemma3. Llama antes a set_prompt().")

        inputs = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        self.response = decoded

    # ---------------- LLaVA (1.5 y 1.6): inferencia específica ----------------

    def _run_llava_inference(self, max_new_tokens: int):
        if self.messages is None:
            raise RuntimeError("No hay messages definidos para LLaVA. Llama antes a set_prompt().")

        prompt = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
        )

        if self.text_only:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            if self.image is None:
                raise RuntimeError("No hay imagen cargada para LLaVA. Llama antes a set_prompt().")
            inputs = self.processor(
                images=self.image,
                text=prompt,
                return_tensors="pt",
            ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generation = output_ids[0][input_len:]
        self.response = self.processor.decode(generation, skip_special_tokens=True)

    # ---------------- Ministral 3: inferencia específica ----------------

    def _run_ministral3_inference(self, max_new_tokens: int):
        if self.messages is None:
            raise RuntimeError(
                "No hay messages definidos para Ministral 3. "
                "Llama antes a set_prompt()."
            )

        inputs = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device

        # Mover todos los tensores al dispositivo del modelo.
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if key == "pixel_values":
                    inputs[key] = value.to(
                        device=model_device,
                        dtype=torch.bfloat16,
                    )
                else:
                    inputs[key] = value.to(model_device)

        if not self.text_only and "pixel_values" not in inputs:
            raise RuntimeError(
                "Ministral no procesó la imagen. "
                f"Claves recibidas: {list(inputs.keys())}"
            )

        input_len = inputs["input_ids"].shape[-1]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.do_sample,
            "use_cache": True,
        }

        if self.do_sample:
            generation_kwargs.update(
                temperature=float(self.temperature),
                top_p=float(self.top_p),
            )

            if self.top_k:
                generation_kwargs["top_k"] = int(self.top_k)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_ids = output_ids[0, input_len:]

        self.response = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
    # ---------------- InternVL: inferencia específica ----------------

    def _run_internvl_inference(self, max_new_tokens: int):
        if not self.text_only and self.pixel_values is None:
            raise RuntimeError("InternVL no tiene pixel_values preparados. Llama antes a set_prompt().")

        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            temperature=float(self.temperature),
            top_p=float(self.top_p),
        )
        question = self.prompt if self.text_only else f"<image>\n{self.prompt}"
        response = self.model.chat(
            self.internvl_tokenizer,
            None if self.text_only else self.pixel_values,
            question,
            generation_config,
        )
        self.response = response

    # ---------------------------------------------------------
    #  RUN INFERENCE (entry point)
    # ---------------------------------------------------------

    def run_inference(self, max_new_tokens=256):
        """
        - Gemma3   → apply_chat_template (oficial).
        - Qwen3-VL → apply_chat_template + generate + recorte.
        - LLaVA    → apply_chat_template + imagen.
        - InternVL → model.chat(tokenizer, pixel_values, question, generation_config).
        """
        if self.is_gemma_chat:
            return self._run_gemma3_inference(max_new_tokens)

        if self.is_qwen3_chat:
            return self._run_qwen3_vl_inference(max_new_tokens)

        if self.is_qwen35_text_chat:
            return self._run_qwen35_inference(max_new_tokens)

        if self.is_llava_chat:
            return self._run_llava_inference(max_new_tokens)

        if self.is_internvl:
            return self._run_internvl_inference(max_new_tokens)
        
        if self.is_ministral3:
            return self._run_ministral3_inference(max_new_tokens)
        
        raise RuntimeError("Configuración de VLM no reconocida para run_inference().")

    def _remove_thinking_content(self, text):
        if not isinstance(text, str) or not text:
            return text

        # Remove common chain-of-thought wrappers used by some chat models.
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    def get_response(self):
        return self._remove_thinking_content(self.response)
