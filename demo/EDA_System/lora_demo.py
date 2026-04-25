"""
LoRA Fine-tuned Qwen2.5-VL-7B Inference Module
Loads the locally trained LoRA adapter for multilabel EDA schematic function classification.

Labels: power, communication, interface, control, signal, eval_board
Adapter: D:/FYP/Classifier/paper_results/FYP-main/task2_qwen_vl_lora/best_adapter/
"""

import json
import time
from pathlib import Path
import os

# 使用本地缓存，避免每次加载都请求 HuggingFace
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_ADAPTER_CANDIDATES = [
    # 键名已修复的 adapter（兼容 transformers 4.49 + peft 0.18）
    "D:/FYP/Classifier/paper_results/task2_qwen_vl_lora/best_adapter_fixed",
    # 原始权重（键名可能与当前 peft 版本不兼容）
    "D:/FYP/Classifier/paper_results/task2_qwen_vl_lora/best_adapter",
]


def _resolve_adapter_path() -> str:
    env_path = os.getenv("LORA_ADAPTER_PATH", "").strip()
    if env_path:
        return env_path
    for p in _ADAPTER_CANDIDATES:
        if Path(p).exists():
            return p
    return _ADAPTER_CANDIDATES[0]


ADAPTER_PATH = _resolve_adapter_path()

_DEFAULT_CLASSES = ["power", "communication", "interface", "control", "signal"]
_classes_env = os.getenv("LORA_CLASSES", "").strip()
if _classes_env:
    CLASSES = [x.strip().lower() for x in _classes_env.split(",") if x.strip()]
else:
    CLASSES = _DEFAULT_CLASSES

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

CLASS_DESCRIPTIONS = {
    "power": "Power Management (PMIC, LDO, Buck/Boost)",
    "communication": "Communication (WiFi, BT, Zigbee, RF)",
    "interface": "Interface (USB, UART, CAN, LIN, SPI, I2C)",
    "control": "Control (MCU, FPGA, Motor Driver)",
    "signal": "Signal Processing (ADC, DAC, Op-Amp, Filter)",
}

# Prompts must match evaluate_gold_test.py (the script that produced the 56.7% EM benchmark).
# System prompt is present; USER_PROMPT asks for comma-separated output (matching training format).
SYSTEM_PROMPT = (
    "You are an expert EDA schematic function classifier. "
    "Analyze the provided schematic image and strictly output the corresponding functional categories. "
    "Rely ONLY on visible circuit evidence (e.g., components, topology, interfaces) and avoid guessing "
    "based solely on background knowledge."
)

USER_PROMPT = (
    "Analyze the schematic diagram and identify its functional categories. "
    "The categories must be chosen from the following list ONLY: "
    "[power, interface, communication, signal, control]. "
    "Output the identified categories separated by commas, with no other text."
)


def _parse_predicted_labels(raw_text: str):
    text = (raw_text or "").strip()
    if not text:
        return []
    text = text.replace("<tool_call>", " ").replace("</tool_call>", " ").strip()

    # Primary: comma-separated labels (matches training output format)
    labels = []
    for token in text.split(","):
        s = token.strip().strip("'\"").lower()
        if s in CLASS_TO_IDX and s not in labels:
            labels.append(s)
    if labels:
        return labels

    # Fallback: JSON array format
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            cleaned = []
            for item in obj:
                s = str(item).strip().lower()
                if s in CLASS_TO_IDX and s not in cleaned:
                    cleaned.append(s)
            return cleaned
    except Exception:
        pass
    if "[" in text and "]" in text:
        chunk = text[text.find("["):text.find("]") + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, list):
                return [str(item).strip().lower() for item in obj if str(item).strip().lower() in CLASS_TO_IDX]
        except Exception:
            pass
    # Fallback 2: keyword presence.
    lowered = text.lower()
    return [c for c in CLASSES if c in lowered]


class LoRADemoSystem:
    def __init__(self):
        self.model = None
        self.processor = None
        self.enabled = False
        self._load_error = None
        self.bad_words_ids = None
        # Lazy load: model is NOT loaded at init.
        # Call ensure_loaded() before inference.

    def ensure_loaded(self, progress_cb=None):
        """Load model if not yet loaded. progress_cb(msg) is called with status strings."""
        if self.enabled:
            return True
        if self._load_error and "Adapter not found" in self._load_error:
            return False  # unrecoverable
        return self._try_load(progress_cb=progress_cb)

    def _try_load(self, progress_cb=None):
        def _prog(msg):
            print(msg)
            if progress_cb:
                progress_cb(msg)

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from peft import PeftModel
        except ImportError as e:
            self._load_error = f"Missing dependency: {e}. Install: pip install transformers peft qwen-vl-utils"
            _prog(f"[LoRADemo] {self._load_error}")
            return False

        adapter_dir = Path(ADAPTER_PATH)
        if not adapter_dir.exists():
            self._load_error = f"Adapter not found: {ADAPTER_PATH}"
            _prog(f"[LoRADemo] {self._load_error}")
            return False

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from peft import PeftModel

            _prog("[LoRADemo] Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True,
                use_fast=False,
                min_pixels=256*28*28,
                max_pixels=512*28*28,
            )

            # Prevent tool-calling control tags from being generated in plain classification mode.
            try:
                tokenizer = self.processor.tokenizer
                bad_words_ids = []
                for token_text in ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]:
                    # Try as regular text encoding first
                    token_ids = tokenizer.encode(token_text, add_special_tokens=False)
                    if token_ids:
                        bad_words_ids.append(token_ids)
                    # Also check special token vocab directly
                    if hasattr(tokenizer, 'added_tokens_encoder'):
                        tid = tokenizer.added_tokens_encoder.get(token_text)
                        if tid is not None:
                            entry = [tid]
                            if entry not in bad_words_ids:
                                bad_words_ids.append(entry)
                self.bad_words_ids = bad_words_ids if bad_words_ids else None
                print(f"[LoRADemo] bad_words_ids: {bad_words_ids}")
            except Exception as e:
                print(f"[LoRADemo] bad_words_ids setup failed: {e}")
                self.bad_words_ids = None

            _prog("[LoRADemo] Loading base model in bf16...")
            has_cuda = torch.cuda.is_available()

            _offload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "offload_tmp")
            os.makedirs(_offload_dir, exist_ok=True)

            if has_cuda:
                _prog("[LoRADemo] Loading bf16 with auto device map...")
                base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_memory={0: "15GiB", "cpu": "12GiB"},
                    offload_folder=_offload_dir,
                )
            else:
                base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

            _prog("[LoRADemo] Applying LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                base_model, str(adapter_dir), offload_folder=_offload_dir
            )
            self.model.eval()

            self.enabled = True
            _prog("[LoRADemo] LoRA model loaded successfully.")
            return True
        except Exception as e:
            import traceback
            self._load_error = str(e)
            traceback.print_exc()
            _prog(f"[LoRADemo] Failed to load model: {e}")
            return False

    def analyze(self, image_path: str) -> dict:
        if not self.enabled:
            return {
                "status": "failed",
                "error": self._load_error or "LoRA model not available",
            }

        try:
            import torch
            from qwen_vl_utils import process_vision_info

            abs_path = str(Path(image_path).absolute())

            # NOTE: prompt format matches evaluate_gold_test.py (the gold benchmark script).
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": abs_path,
                         "min_pixels": 4 * 28 * 28,
                         "max_pixels": 200704},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ]

            start_time = time.time()

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                tools=None,  # explicitly disable tool-calling mode
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # With device_map="auto", send inputs to cuda:0
            target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

            n_tokens = inputs["input_ids"].shape[1]
            print(f"[LoRADemo] Input tokens: {n_tokens}, generating...")
            gen_start = time.time()

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    min_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                    bad_words_ids=self.bad_words_ids,
                )

            gen_time = time.time() - gen_start
            print(f"[LoRADemo] Generation took {gen_time:.1f}s")

            in_len = inputs["input_ids"].shape[1]
            generated_ids_trimmed = generated_ids[:, in_len:]
            print(f"[LoRADemo] generated {generated_ids_trimmed.shape[1]} new tokens; first ids={generated_ids_trimmed[0].tolist()[:10]}")
            raw_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            print(f"[LoRADemo] raw_text={repr(raw_text)}")

            elapsed = time.time() - start_time
            labels = _parse_predicted_labels(raw_text)

            label_details = [
                {
                    "label": lbl,
                    "description": CLASS_DESCRIPTIONS.get(lbl, lbl),
                }
                for lbl in labels
            ]

            return {
                "status": "success",
                "labels": labels,
                "label_details": label_details,
                "raw_output": raw_text,
                "inference_time": elapsed,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}
