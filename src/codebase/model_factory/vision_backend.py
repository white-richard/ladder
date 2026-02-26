from pathlib import Path
from typing import Optional, Sequence

import clip
import torch
from PIL import Image


def _is_mammo_dataset(dataset: str) -> bool:
    return str(dataset).lower() in {"rsna", "vindr", "embed", "cbis", "cbis-ddsm"}


def _validate_llava_checkpoint_path(llava_path: Path) -> None:
    if not llava_path.exists():
        raise FileNotFoundError(f"Llava model path does not exist: {llava_path}")

    if not llava_path.is_dir():
        raise ValueError(
            "--llava_model_path must point to an unpacked checkpoint directory, not a file. "
            f"Got: {llava_path}"
        )

    has_adapter_pair = (llava_path / "adapter_config.json").exists() and (
        llava_path / "adapter_model.safetensors"
    ).exists()
    has_full_checkpoint = (llava_path / "config.json").exists()
    if not has_adapter_pair and not has_full_checkpoint:
        raise ValueError(
            "Invalid Llava checkpoint directory. Expected either: "
            "(1) adapter_config.json + adapter_model.safetensors, or "
            "(2) a merged/full checkpoint containing config.json. "
            f"Directory: {llava_path}"
        )


def resolve_backend_tag(args) -> str:
    backend = getattr(args, "backend", "legacy")
    if backend == "llava_mammo":
        return "llava_mammo"
    return str(getattr(args, "clip_vision_encoder", "legacy"))


def resolve_backend_save_path(args) -> Path:
    return Path(args.save_path.format(args.seed)) / f"clip_img_encoder_{resolve_backend_tag(args)}"


def validate_backend_args(args) -> None:
    backend = getattr(args, "backend", "legacy")
    if backend not in {"legacy", "llava_mammo"}:
        raise ValueError(f"Unsupported backend: {backend}")

    if backend == "llava_mammo":
        if not _is_mammo_dataset(getattr(args, "dataset", "")):
            raise ValueError(
                "llava_mammo backend is currently supported only for mammography datasets "
                "(RSNA, VinDr, CBIS, Embed)."
            )
        model_path = getattr(args, "llava_model_path", "")
        if not model_path:
            raise ValueError("--llava_model_path is required when --backend llava_mammo is selected.")
        _validate_llava_checkpoint_path(Path(model_path))


def _normalize_texts(texts):
    if isinstance(texts, str):
        return [texts]
    return list(texts)


class LegacyEmbeddingBackend:
    name = "legacy"

    def __init__(self, clip_model: dict, device: str = "cuda"):
        self.clip_model = clip_model
        self.device = device

    def encode_images(
            self, img: torch.Tensor, dataset: str, image_paths: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        dataset = str(dataset).lower()
        if dataset in {"waterbirds", "celeba", "metashift"}:
            reps = self.clip_model["model"].encode_image(img)
            reps = reps / reps.norm(dim=-1, keepdim=True)
            return reps
        if _is_mammo_dataset(dataset):
            return self.clip_model["model"].encode_image_normalized(img)
        if dataset == "nih" and self.clip_model.get("type") == "cxr_clip":
            reps = self.clip_model["model"].encode_image(img)
            reps = (
                self.clip_model["model"].image_projection(reps)
                if self.clip_model["model"].projection
                else reps
            )
            reps = reps / torch.norm(reps, dim=1, keepdim=True)
            return reps

        if hasattr(self.clip_model["model"], "encode_image_normalized"):
            return self.clip_model["model"].encode_image_normalized(img)
        reps = self.clip_model["model"].encode_image(img)
        return reps / reps.norm(dim=-1, keepdim=True)

    def encode_texts(self, texts, dataset_type: str = "medical") -> torch.Tensor:
        texts = _normalize_texts(texts)
        if len(texts) == 0:
            raise ValueError("No text prompts were provided for embedding.")

        with torch.no_grad():
            if dataset_type == "medical" and self.clip_model.get("tokenizer") is not None:
                text_token = self.clip_model["tokenizer"](
                    texts, padding="longest", truncation=True, return_tensors="pt", max_length=256
                )
                text_emb = self.clip_model["model"].encode_text(text_token.to(self.device))
                text_emb = (
                    self.clip_model["model"].text_projection(text_emb)
                    if getattr(self.clip_model["model"], "projection", False)
                    else text_emb
                )
            else:
                text_token = clip.tokenize(texts).to(self.device)
                text_emb = self.clip_model["model"].encode_text(text_token)

            text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
            return text_emb.detach()


class LlavaMammoEmbeddingBackend:
    name = "llava_mammo"

    def __init__(self, args):
        self.device = getattr(args, "device", "cuda")
        self.llava_model_path = Path(args.llava_model_path)
        _validate_llava_checkpoint_path(self.llava_model_path)
        self.llava_text_batch_size = int(getattr(args, "llava_text_batch_size", 32))
        if self.llava_text_batch_size < 1:
            raise ValueError("--llava_text_batch_size must be >= 1.")
        self.llava_text_max_length = int(getattr(args, "llava_text_max_length", 256))
        if self.llava_text_max_length < 1:
            raise ValueError("--llava_text_max_length must be >= 1.")
        self.llava_base_model_id = getattr(
            args, "llava_base_model_id", "llava-hf/llava-v1.6-vicuna-7b-hf"
        )
        self.llava_processor_id = getattr(
            args, "llava_processor_id", "llava-hf/llava-v1.6-vicuna-7b-hf"
        )

        try:
            import transformers
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        except Exception as exc:
            raise ImportError(
                "llava_mammo backend requires transformers with LlavaNext support. "
                "Install a compatible transformers version (for example >= 4.44)."
            ) from exc
        if not hasattr(LlavaNextForConditionalGeneration, "get_image_features"):
            raise RuntimeError(
                "The installed transformers build does not expose "
                "`LlavaNextForConditionalGeneration.get_image_features`, so llava_mammo cannot run. "
                f"Detected transformers version: {transformers.__version__}. "
                "Install a transformers release that provides this API."
            )

        dtype = torch.float16 if "cuda" in self.device else torch.float32
        try:
            self.processor = LlavaNextProcessor.from_pretrained(self.llava_processor_id)
        except ValueError as exc:
            if "sentencepiece" in str(exc).lower():
                raise RuntimeError(
                    "Failed to initialize Llava tokenizer. Install sentencepiece in the active environment "
                    "(for example: `uv add sentencepiece`) and retry."
                ) from exc
            raise

        adapter_config = self.llava_model_path / "adapter_config.json"
        adapter_weights = self.llava_model_path / "adapter_model.safetensors"
        if adapter_config.exists() and adapter_weights.exists():
            try:
                from peft import PeftModel
            except Exception as exc:
                raise ImportError(
                    "Detected LoRA adapter checkpoint at --llava_model_path, but peft is not installed. "
                    "Install peft or provide a merged/full Llava checkpoint directory."
                ) from exc

            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                self.llava_base_model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            peft_model = PeftModel.from_pretrained(base_model, str(self.llava_model_path))
            if hasattr(peft_model, "merge_and_unload"):
                self.model = peft_model.merge_and_unload()
            else:
                self.model = peft_model
        else:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                str(self.llava_model_path),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )

        self.model = self.model.to(self.device)
        self.model.eval()
        if getattr(self.model, "get_image_features", None) is None:
            raise RuntimeError(
                "Loaded Llava model does not expose `get_image_features`; refusing to run without a native image "
                f"feature API. Model class: {self.model.__class__.__name__}, transformers version: {transformers.__version__}. "
                "Use a Llava/transformers build that provides `get_image_features`."
            )

    @staticmethod
    def _extract_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                tensor = LlavaMammoEmbeddingBackend._extract_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(obj, dict):
            for item in obj.values():
                tensor = LlavaMammoEmbeddingBackend._extract_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _get_image_features(self, pixel_values: torch.Tensor, image_sizes):
        feature_fn = getattr(self.model, "get_image_features", None)
        if feature_fn is None:
            raise RuntimeError(
                "Llava backend requires model.get_image_features, but it is missing. "
                f"Loaded model class: {self.model.__class__.__name__}."
            )
        vision_feature_layer = getattr(self.model.config, "vision_feature_layer", None)
        vision_feature_select_strategy = getattr(
            self.model.config, "vision_feature_select_strategy", None
        )
        if vision_feature_layer is None or vision_feature_select_strategy is None:
            raise RuntimeError(
                "Llava config is missing vision feature extraction settings "
                "(`vision_feature_layer` / `vision_feature_select_strategy`)."
            )
        try:
            output = feature_fn(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to compute Llava image features via model.get_image_features("
                "pixel_values=..., image_sizes=..., vision_feature_layer=..., "
                "vision_feature_select_strategy=...)."
            ) from exc

        if isinstance(output, (list, tuple)):
            pooled = []
            for idx, feat in enumerate(output):
                if not isinstance(feat, torch.Tensor):
                    raise RuntimeError(
                        f"model.get_image_features returned non-tensor item at index {idx}: {type(feat)}."
                    )
                if feat.ndim < 2:
                    raise RuntimeError(
                        f"model.get_image_features returned invalid tensor rank at index {idx}: shape={tuple(feat.shape)}."
                    )
                reduce_dims = tuple(range(feat.ndim - 1))
                pooled.append(feat.mean(dim=reduce_dims))
            if len(pooled) == 0:
                raise RuntimeError("model.get_image_features returned an empty list; expected one tensor per image.")
            return torch.stack(pooled, dim=0)

        tensor = self._extract_tensor(output)
        if tensor is None:
            raise RuntimeError(
                "model.get_image_features returned an unsupported output type; no tensor could be extracted."
            )
        return tensor

    def encode_images(
            self, img: torch.Tensor, dataset: str, image_paths: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        if image_paths is None:
            raise ValueError(
                "llava_mammo backend requires image paths for image embedding. "
                "Ensure dataloader batches provide 'img_path'."
            )

        images = []
        for path in image_paths:
            with Image.open(str(path)) as image:
                images.append(image.convert("RGB"))
        proc = self.processor(images=images, text=[" "] * len(images), return_tensors="pt", padding=True)
        pixel_values = proc.get("pixel_values")
        image_sizes = proc.get("image_sizes")
        if pixel_values is None:
            raise RuntimeError(
                "Llava processor did not return `pixel_values`; cannot compute Llava image embeddings."
            )
        if image_sizes is None:
            raise RuntimeError(
                "Llava processor did not return `image_sizes`; cannot compute Llava image embeddings."
            )

        pixel_values = pixel_values.to(self.device)
        if isinstance(image_sizes, torch.Tensor):
            image_sizes = image_sizes.to(self.device)

        with torch.no_grad():
            image_features = self._get_image_features(pixel_values, image_sizes)
            if image_features.ndim == 2:
                image_features = image_features.unsqueeze(1)

            target_dim = getattr(getattr(self.model.config, "text_config", None), "hidden_size", None)
            if (
                    target_dim is not None
                    and image_features.size(-1) != target_dim
                    and hasattr(self.model, "multi_modal_projector")
            ):
                image_features = self.model.multi_modal_projector(image_features)

            if image_features.ndim > 2:
                reduce_dims = tuple(range(1, image_features.ndim - 1))
                image_features = image_features.mean(dim=reduce_dims)

            if image_features.ndim == 1:
                image_features = image_features.unsqueeze(0)

            image_features = image_features / torch.norm(image_features, dim=1, keepdim=True)
            return image_features.detach()

    def encode_texts(self, texts, dataset_type: str = "medical") -> torch.Tensor:
        texts = _normalize_texts(texts)
        if len(texts) == 0:
            raise ValueError("No text prompts were provided for embedding.")

        language_model = getattr(self.model, "language_model", None)
        if language_model is None and hasattr(self.model, "get_model"):
            language_model = getattr(self.model.get_model(), "language_model", None)
        if language_model is None:
            raise RuntimeError(
                "Llava model does not expose `language_model`; cannot compute Llava text embeddings."
            )

        text_backbone = getattr(language_model, "model", None)
        if text_backbone is None:
            raise RuntimeError(
                "Llava language_model does not expose `.model`; cannot compute memory-safe text embeddings."
            )

        tokenizer = self.processor.tokenizer
        text_emb_batches = []
        for start in range(0, len(texts), self.llava_text_batch_size):
            chunk = texts[start:start + self.llava_text_batch_size]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.llava_text_max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = text_backbone(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_dict=True,
                    output_hidden_states=False,
                    use_cache=False,
                )
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    raise RuntimeError(
                        "Llava text backbone output does not contain last_hidden_state."
                    )

                attn_mask = encoded["attention_mask"].unsqueeze(-1).float()
                text_emb = (hidden * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp_min(1.0)
                text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
                text_emb_batches.append(text_emb.detach())

        return torch.cat(text_emb_batches, dim=0)
