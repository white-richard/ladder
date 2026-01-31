import torch
from torch import nn

from . import EfficientNet
from .classifier import LinearClassifier


class MammoClassifier(nn.Module):
    def __init__(self, arch, clf_checkpoint, n_class):
        super(MammoClassifier, self).__init__()
        self.clf = EfficientNet.from_pretrained(arch, num_classes=n_class)
        self.ckpt = torch.load(clf_checkpoint, map_location="cpu")
        ckpt_state = self.ckpt.get("model", self.ckpt)

        def _normalize_key(key):
            prefixes = ("module.", "image_encoder.")
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        changed = True
            return key

        def _filter_state_dict(state_dict, target_state_dict):
            filtered = {}
            mismatched = {}
            for key, value in state_dict.items():
                if key not in target_state_dict:
                    continue
                if target_state_dict[key].shape != value.shape:
                    mismatched[key] = (tuple(value.shape), tuple(target_state_dict[key].shape))
                    continue
                filtered[key] = value
            return filtered, mismatched

        image_encoder_weights = {}
        if isinstance(ckpt_state, dict):
            for k, v in ckpt_state.items():
                normalized = _normalize_key(k)
                if normalized.startswith("_fc."):
                    continue
                image_encoder_weights[normalized] = v

        image_encoder_weights, enc_mismatch = _filter_state_dict(
            image_encoder_weights, self.clf.state_dict()
        )
        ret = self.clf.load_state_dict(image_encoder_weights, strict=False)
        print(ret)
        if enc_mismatch:
            print(f"Skipped {len(enc_mismatch)} encoder keys with shape mismatches")

        clf_ft_dim = 0
        if arch.lower() == "efficientnet-b5":
            clf_ft_dim = 2048

        self.classifier = LinearClassifier(feature_dim=clf_ft_dim, num_class=n_class)
        image_clf_weights = {}
        if isinstance(ckpt_state, dict):
            weight_keys = (
                "_fc.weight",
                "image_encoder._fc.weight",
                "module._fc.weight",
                "module.image_encoder._fc.weight",
            )
            bias_keys = (
                "_fc.bias",
                "image_encoder._fc.bias",
                "module._fc.bias",
                "module.image_encoder._fc.bias",
            )
            for k in weight_keys:
                if k in ckpt_state:
                    image_clf_weights["classification_head.weight"] = ckpt_state[k]
                    break
            for k in bias_keys:
                if k in ckpt_state:
                    image_clf_weights["classification_head.bias"] = ckpt_state[k]
                    break

        if image_clf_weights:
            image_clf_weights, clf_mismatch = _filter_state_dict(
                image_clf_weights, self.classifier.state_dict()
            )
            if image_clf_weights:
                ret = self.classifier.load_state_dict(image_clf_weights, strict=False)
                print(ret)
            else:
                clf_mismatch_count = len(clf_mismatch)
                if clf_mismatch_count:
                    print(f"Skipped {clf_mismatch_count} classifier keys with shape mismatches")
        else:
            print("No classifier head weights found in checkpoint; using random init.")

    def get_predictions_from_chkpt(self):
        return self.ckpt["predictions"]

    def forward(self, images):
        features = self.clf(images)
        logits = self.classifier(features)
        return features, logits
