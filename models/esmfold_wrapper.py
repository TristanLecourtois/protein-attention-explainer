"""
ESMFold wrapper with attention map extraction via forward hooks.

ESMFold = ESM-2 (language model trunk) + folding head.
We capture attention weights from every ESM encoder layer
using PyTorch forward hooks — fully compatible with HuggingFace
transformers without modifying the model source.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ESMFoldWrapper:
    """
    Wraps facebook/esmfold_v1 and exposes:
      - PDB string prediction
      - Per-layer, per-head attention tensors [L, H, N, N]
    """

    def __init__(
        self,
        model_name: str = "facebook/esmfold_v1",
        device: Optional[str] = None,
        use_half: bool = True,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.use_half = use_half and self.device != "cpu"

        self._attention_cache: list[torch.Tensor] = []
        self._hooks: list = []

        self.model = None
        self.tokenizer = None


    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device and device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self):
        """Lazy-load model (can take 1-2 min on first call — downloads ~2.5 GB)."""
        if self.model is not None:
            return self

        try:
            from transformers import AutoTokenizer, EsmForProteinFolding
        except ImportError as e:
            raise ImportError("pip install transformers>=4.35.0") from e

        logger.info("Loading ESMFold from %s …", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = EsmForProteinFolding.from_pretrained(
            self.model_name, low_cpu_mem_usage=True
        )

        if self.use_half:
            self.model = self.model.half()

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("ESMFold loaded on %s (half=%s)", self.device, self.use_half)
        return self


    def _register_attention_hooks(self):
        """Hook every ESM-2 encoder self-attention layer to grab weights."""
        self._attention_cache.clear()
        self._remove_hooks()

        esm_encoder = self.model.esm.encoder

        for layer in esm_encoder.layer:
            # Each BertLayer has layer.attention.self — output is
            # (context_layer, attention_probs) when output_attentions=True.
            hook = layer.attention.self.register_forward_hook(
                self._make_hook()
            )
            self._hooks.append(hook)

    def _make_hook(self):
        def hook_fn(module, inputs, output):
            # output = (context, attention_probs) or just context
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]  # [batch, heads, N, N]
                self._attention_cache.append(attn.detach().float().cpu())
        return hook_fn

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


    def predict(self, sequence: str) -> dict:
        """
        Run ESMFold on *sequence* and return:
          {
            "pdb": str,
            "attentions": np.ndarray  # [L, H, N, N]  float32
            "plddt": np.ndarray       # [N]  per-residue confidence
            "sequence": str
          }
        """
        self.load()
        sequence = sequence.strip().upper()
        n = len(sequence)
        logger.info("Running ESMFold on sequence length %d", n)

        # Tokenise (no special tokens — ESMFold handles them internally)
        tokenized = self.tokenizer(
            [sequence],
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        self._register_attention_hooks()

        with torch.no_grad():
            outputs = self.model(**tokenized)

        self._remove_hooks()

        pdb_string = self.model.output_to_pdb(outputs)[0]

        plddt = outputs.plddt[0].float().cpu().numpy()  # [N, 37] → mean over atoms
        if plddt.ndim == 2:
            plddt = plddt.mean(axis=-1)

        if self._attention_cache:
            # Stack → [L, batch=1, H, N+2, N+2]  (includes CLS/EOS tokens)
            stacked = torch.stack(self._attention_cache, dim=0)  # [L, 1, H, N+2, N+2]
            stacked = stacked[:, 0, :, 1:-1, 1:-1]  # strip special tokens → [L, H, N, N]
            attentions = stacked.numpy()  # float32
        else:
            logger.warning("No attention weights captured — returning zeros.")
            num_layers = len(self.model.esm.encoder.layer)
            num_heads = self.model.esm.config.num_attention_heads
            attentions = np.zeros((num_layers, num_heads, n, n), dtype=np.float32)

        return {
            "pdb": pdb_string,
            "attentions": attentions,
            "plddt": plddt,
            "sequence": sequence,
        }


    def save_pdb(self, pdb_string: str, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pdb_string)
        return path

    @property
    def num_layers(self) -> int:
        self.load()
        return len(self.model.esm.encoder.layer)

    @property
    def num_heads(self) -> int:
        self.load()
        return self.model.esm.config.num_attention_heads
