"""
Pseudo-labeler M for CoBELa.

Loads the supervised concept classifiers from CB-AE to produce
binary pseudo-labels S_hat = M(x) for training.

File naming convention (from CB-AE):
    checkpoints/celebahq_{ConceptName}_rn18_conclsf.pth

Usage:
    M = PseudoLabeler(
        weights_dir="checkpoints",
        concept_names=["Smiling", "Male", ...],
        arch="resnet18",
        dataset_prefix="celebahq",
    )
    labels = M(images)        # (batch, K) binary
    probs  = M.predict_proba(images)  # (batch, K) float
"""

import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T


class PseudoLabeler(nn.Module):

    def __init__(
        self,
        weights_dir: str,
        concept_names: list,
        arch: str = "resnet18",
        dataset_prefix: str = "celebahq",
        resolution: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.concept_names = concept_names
        self.K = len(concept_names)
        self.device = device

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.classifiers = nn.ModuleList()
        for name in concept_names:
            clf = self._load_classifier(weights_dir, name, arch, dataset_prefix)
            clf.eval()
            for p in clf.parameters():
                p.requires_grad_(False)
            self.classifiers.append(clf)

        self.to(device)
        print(f"[pseudolabeler] Loaded {self.K} classifiers ({arch}) from {weights_dir}")

    def _load_classifier(self, weights_dir, concept_name, arch, dataset_prefix):
        arch_tag = arch.replace("resnet", "rn")
        filename = f"{dataset_prefix}_{concept_name}_{arch_tag}_conclsf.pth"
        ckpt_path = os.path.join(weights_dir, filename)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Classifier not found: {ckpt_path}\n"
                f"Available: {[f for f in os.listdir(weights_dir) if f.endswith('.pth')]}"
            )

        if arch == "resnet18":
            model = tv_models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif arch == "resnet50":
            model = tv_models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        else:
            raise ValueError(f"Unknown arch: {arch}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]

        model.load_state_dict(state, strict=False)
        return model

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2.0
        x = x.clamp(0, 1)
        x = self.normalize(x)
        return x

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        x_prep = self._preprocess(x)
        probs = []
        for clf in self.classifiers:
            logits = clf(x_prep)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.append(p)
        return torch.stack(probs, dim=1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = self.predict_proba(x)
        return (probs >= threshold).long()
