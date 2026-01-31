import torch
from torch import nn

from . import EfficientNet
from .classifier import LinearClassifier


class MammoClassifier(nn.Module):
    def __init__(self, arch, clf_checkpoint, n_class):
        super(MammoClassifier, self).__init__()
        self.clf = EfficientNet.from_pretrained(arch, num_classes=n_class)
        self.ckpt = torch.load(clf_checkpoint, map_location="cpu")

        # Normalize keys (strip a possible 'module.' prefix)
        raw_state_dict = self.ckpt["model"]
        if any(k.startswith("module.") for k in raw_state_dict):
            raw_state_dict = {k.replace("module.", "", 1): v for k, v in raw_state_dict.items()}

        # Identify classification head keys (support multiple naming conventions)
        fc_weight_key = next(
            (k for k in raw_state_dict if k.endswith("_fc.weight") or k.endswith("classifier.weight")
             or k.endswith("classification_head.weight")),
            None,
        )
        fc_bias_key = next(
            (k for k in raw_state_dict if k.endswith("_fc.bias") or k.endswith("classifier.bias")
             or k.endswith("classification_head.bias")),
            None,
        )

        # Keep only keys that exist in the EfficientNet backbone
        backbone_keys = set(self.clf.state_dict().keys())
        image_encoder_weights = {
            k: v for k, v in raw_state_dict.items()
            if k in backbone_keys and k not in {fc_weight_key, fc_bias_key}
        }

        ret = self.clf.load_state_dict(image_encoder_weights, strict=False)
        print("Loaded encoder weights:", ret)

        clf_ft_dim = 2048 if arch.lower() == "efficientnet-b5" else 0
        self.classifier = LinearClassifier(feature_dim=clf_ft_dim, num_class=n_class)

        image_clf_weights = {}
        if fc_weight_key:
            image_clf_weights["classification_head.weight"] = raw_state_dict[fc_weight_key]
        if fc_bias_key:
            image_clf_weights["classification_head.bias"] = raw_state_dict[fc_bias_key]

        ret = self.classifier.load_state_dict(image_clf_weights, strict=False)
        print("Loaded head weights:", ret)

    def get_predictions_from_chkpt(self):
        return self.ckpt["predictions"]

    def forward(self, images):
        features = self.clf(images)
        logits = self.classifier(features)
        return features, logits

# import torch
# from torch import nn

# from . import EfficientNet
# from .classifier import LinearClassifier


# class MammoClassifier(nn.Module):
#     def __init__(self, arch, clf_checkpoint, n_class):
#         super(MammoClassifier, self).__init__()
#         self.clf = EfficientNet.from_pretrained(arch, num_classes=n_class)
#         self.ckpt = torch.load(clf_checkpoint, map_location="cpu")
#         image_encoder_weights = {}
#         for k in self.ckpt["model"].keys():
#             image_encoder_weights[k] = self.ckpt["model"][k]

#         image_encoder_weights.pop("_fc.weight")
#         image_encoder_weights.pop("_fc.bias")
#         ret = self.clf.load_state_dict(image_encoder_weights, strict=True)
#         print(ret)

#         clf_ft_dim = 0
#         if arch.lower() == "efficientnet-b5":
#             clf_ft_dim = 2048

#         self.classifier = LinearClassifier(feature_dim=clf_ft_dim, num_class=n_class)
#         image_clf_weights = {}
#         for k in self.ckpt["model"].keys():
#             if k == "_fc.weight":
#                 image_clf_weights["classification_head.weight"] = self.ckpt["model"][k]
#             elif k == "_fc.bias":
#                 image_clf_weights["classification_head.bias"] = self.ckpt["model"][k]
#         ret = self.classifier.load_state_dict(image_clf_weights, strict=True)
#         print(ret)

#     def get_predictions_from_chkpt(self):
#         return self.ckpt["predictions"]

#     def forward(self, images):
#         features = self.clf(images)
#         logits = self.classifier(features)
#         return features, logits
