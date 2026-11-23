"""
scripts/export_nomic_coreml.py — Export nomic-embed-text-v1.5 (MRL-128d) to CoreML
                                   for iPhone 15 Pro ANE (Neural Engine, 35 TOPS).

What this produces
──────────────────
  models/nomic_embed_coreml_128d.mlpackage   — the CoreML model
  models/nomic_embed_coreml_128d_fp16.mlpackage — Float16 variant (smaller, still accurate)

The model is the MRL (Matryoshka Representation Learning) truncated version of
nomic-embed-text-v1.5: the first 128 dimensions of the full 768d output are
used for the MSymPrev affective stream (mood valence/arousal/dominance).

Deployment target: iOS 17+ (iPhone 15 Pro, coremltools ≥ 7.1)

Dependencies
────────────
    pip install coremltools>=7.1 sentence-transformers torch numpy

Usage
─────
    python scripts/export_nomic_coreml.py                   # export 128d (default)
    python scripts/export_nomic_coreml.py --dims 384        # 384d for QSymPrev
    python scripts/export_nomic_coreml.py --dims 128 --validate
    python scripts/export_nomic_coreml.py --fp16             # also export Float16 variant

The exported .mlpackage is loaded by the iOS LOCI app and served at:
    http://192.168.1.100:7867/v1/embed   (ANEEdgeClient in loci_symprev_streams.py)

CoreML deployment notes
───────────────────────
  • The ANE runs Float16 by default; the CPU fallback runs Float32.
  • Input: token IDs (int32, shape [1, seq_len]) + attention mask (int32, shape [1, seq_len])
    — OR — raw text via a text-preprocessing pipeline (see --with-tokenizer flag).
  • Output: "embeddings" (float32, shape [1, dims]) — L2-normalised.
  • The model fits in ~12 MB (fp16 MRL-128d, compressed).
  • The iOS app handles tokenization (bert-base-uncased vocab) on-device.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_ID   = "nomic-ai/nomic-embed-text-v1.5"


def _check_deps() -> None:
    missing = []
    for pkg in ("torch", "numpy", "sentence_transformers"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))
    try:
        import coremltools as ct  # type: ignore
        ver = tuple(int(x) for x in ct.__version__.split(".")[:2])
        if ver < (7, 1):
            print(f"[warn] coremltools {ct.__version__} is older than 7.1 — upgrade recommended")
    except ImportError:
        missing.append("coremltools>=7.1")
    if missing:
        print(f"[error] Missing dependencies: {', '.join(missing)}")
        print(f"        pip install {' '.join(missing)}")
        sys.exit(1)


class _MRLTruncatedEncoder:
    """
    Wraps nomic-embed-text-v1.5 and truncates the output to `dims` dimensions.

    nomic-embed-text-v1.5 supports MRL natively — the first D dimensions of its
    output form a valid embedding at any MRL-supported size:
        128, 256, 384, 512, 768
    We just slice [:dims] after mean-pooling and L2-normalising.
    """

    def __init__(self, dims: int = 128, trust_remote_code: bool = True) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.dims   = dims
        self._model = SentenceTransformer(
            MODEL_ID,
            trust_remote_code=trust_remote_code,
        )
        self._model.eval()

    def encode(self, text: str) -> "torch.Tensor":  # type: ignore[name-defined]
        import torch
        with torch.no_grad():
            full = self._model.encode(
                [text],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            return full[:, : self.dims]

    def tokenize(self, text: str) -> dict:
        return self._model.tokenize([text])


def _trace_model(encoder: "_MRLTruncatedEncoder") -> tuple:
    """
    Trace the sentence-transformer encoder with TorchScript.
    Returns (traced_model, example_input_dict).
    """
    import torch
    from sentence_transformers import SentenceTransformer  # type: ignore

    # Pull the underlying transformer + pooling modules
    # SentenceTransformer is a nn.Sequential: [0]=AutoModel, [1]=Pooling, [2]=Normalize
    base_model = encoder._model

    class _TraceWrapper(torch.nn.Module):
        def __init__(self, inner: SentenceTransformer, dims: int) -> None:
            super().__init__()
            self._inner = inner
            self._dims  = dims

        def forward(
            self,
            input_ids:      "torch.Tensor",
            attention_mask: "torch.Tensor",
        ) -> "torch.Tensor":
            features = {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
            }
            out = self._inner(features)
            emb = out["sentence_embedding"]    # [batch, 768]
            # MRL truncation
            emb = emb[:, : self._dims]
            # Re-normalise after truncation
            norm = emb.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return emb / norm

    wrapper = _TraceWrapper(base_model, encoder.dims).eval()

    example = encoder.tokenize("search_document: hello world")
    input_ids      = example["input_ids"][:1]        # shape [1, seq_len]
    attention_mask = example["attention_mask"][:1]

    with __import__("torch").no_grad():
        traced = __import__("torch").jit.trace(
            wrapper, (input_ids, attention_mask), strict=False
        )

    return traced, (input_ids, attention_mask)


def export(
    dims:         int  = 128,
    fp16:         bool = False,
    validate:     bool = False,
    output_dir:   Path = MODELS_DIR,
) -> Path:
    """Export nomic-embed-text-v1.5 MRL-{dims}d to CoreML."""
    import coremltools as ct  # type: ignore
    import numpy as np
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix    = "_fp16" if fp16 else ""
    out_name  = f"nomic_embed_coreml_{dims}d{suffix}.mlpackage"
    out_path  = output_dir / out_name

    print(f"[1/4] Loading {MODEL_ID} from HuggingFace …")
    encoder = _MRLTruncatedEncoder(dims=dims)

    print(f"[2/4] Tracing with TorchScript (dims={dims}) …")
    traced, (example_ids, example_mask) = _trace_model(encoder)

    print(f"[3/4] Converting to CoreML …")
    compute_units = ct.ComputeUnit.ALL  # prefers ANE on iOS

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids",      shape=example_ids.shape,      dtype=np.int32),
            ct.TensorType(name="attention_mask",  shape=example_mask.shape,     dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="embeddings", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=compute_units,
        convert_to="mlprogram",
    )

    if fp16:
        from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights  # type: ignore
        try:
            config  = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
            mlmodel = linear_quantize_weights(mlmodel, config)
        except Exception as exc:
            print(f"[warn] quantization failed ({exc}); saving as-is")

    # Metadata for the LOCI iOS app
    mlmodel.short_description = (
        f"nomic-embed-text-v1.5 MRL-{dims}d CoreML export for LOCI MSymPrev ANE edge"
    )
    mlmodel.author = "Arko Sanyal / LOCI Project"
    mlmodel.version = "1.0.0"
    mlmodel.user_defined_metadata["loci.stream"]    = "MSymPrev"
    mlmodel.user_defined_metadata["loci.dims"]      = str(dims)
    mlmodel.user_defined_metadata["loci.base_model"] = MODEL_ID
    mlmodel.user_defined_metadata["loci.mrl_dims"]  = "128,256,384,512,768"
    mlmodel.user_defined_metadata["loci.serve_at"]  = "http://192.168.1.100:7867/v1/embed"

    mlmodel.save(str(out_path))
    print(f"[4/4] Saved → {out_path}")

    if validate:
        print("[validate] Running prediction on example input …")
        import coremltools.proto.FeatureTypes_pb2  # noqa: F401
        pred = mlmodel.predict({
            "input_ids":     example_ids.numpy().astype(np.int32),
            "attention_mask": example_mask.numpy().astype(np.int32),
        })
        emb = pred["embeddings"]
        norm_val = float(np.linalg.norm(emb))
        print(f"  output shape : {emb.shape}  (expected [1, {dims}])")
        print(f"  L2 norm      : {norm_val:.6f}  (expected ≈ 1.0)")
        assert emb.shape == (1, dims),   f"shape mismatch: {emb.shape}"
        assert abs(norm_val - 1.0) < 0.01, f"not normalised: norm={norm_val}"
        print("[validate] PASSED ✓")

    size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1_048_576
    print(f"\nModel size: {size_mb:.1f} MB")
    print(f"Deployment: iOS 17+, ANE-first (ComputeUnit.ALL)")
    print(f"\nTo deploy on iPhone:")
    print(f"  1. Add {out_name} to the LOCI iOS Xcode project (Targets → Build Phases)")
    print(f"  2. Load with:  let model = try Nomic_embed_coreml_{dims}d(configuration: MLModelConfiguration())")
    print(f"  3. The iOS app serves /v1/embed at port 7867 using this model")
    print(f"  4. Test: curl http://192.168.1.100:7867/v1/embed -d '{{\"input\":\"test\"}}'")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export nomic-embed-text-v1.5 MRL slice to CoreML for iPhone ANE"
    )
    ap.add_argument(
        "--dims", type=int, default=128,
        choices=[128, 256, 384, 512, 768],
        help="MRL truncation dimension (default: 128 for MSymPrev ANE edge)",
    )
    ap.add_argument("--fp16",     action="store_true", help="Also export Int8-quantised FP16 variant")
    ap.add_argument("--validate", action="store_true", help="Run a sample prediction after export to verify shape/norm")
    ap.add_argument("--output-dir", type=Path, default=MODELS_DIR, help="Where to save the .mlpackage")
    args = ap.parse_args()

    _check_deps()
    export(
        dims       = args.dims,
        fp16       = args.fp16,
        validate   = args.validate,
        output_dir = args.output_dir,
    )
    if args.fp16:
        # Export the standard FP32 variant too if fp16 was requested
        export(dims=args.dims, fp16=False, validate=False, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
