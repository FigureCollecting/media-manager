"""Pluggable matting (background-removal) pipeline.

MODEL RULE: this module never downloads a real model or imports
onnxruntime at module scope. StubMattingBackend is a lightweight
luminance/edge threshold against a near-uniform backdrop -- good enough to
exercise the pipeline end-to-end in dev/tests without a GPU. The real
backend (BiRefNetMattingBackend, running BiRefNet via onnxruntime-gpu) is
implemented below, but its ONNX-runtime import and model-session load are
LAZY -- they only happen inside BiRefNetMattingBackend._get_session(), the
first time inference actually runs. Tests exercise the full
preprocess/run/postprocess/apply-alpha pipeline by injecting a fake
session (ctor `session=`) or overriding `_run()`, so onnxruntime and real
model weights are never needed in application code or tests. The weights
themselves + GPU runtime are baked into the worker Docker image at deploy
time (see Dockerfile.worker), keyed by Settings.matting_model_path / env
MATTING_MODEL_PATH.

CRITICAL: nothing in this module (or its callers) may call
``.convert("RGB")`` on the working image -- that silently drops the alpha
channel that the grounding scalars (app.workers.grounding) and the serve
route depend on. Always work in RGBA.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from PIL import Image

from ..hashing import detect_mime

if TYPE_CHECKING:
    from ..config import Settings


class MattingBackend(ABC):
    """Interface for producing an RGBA cutout from a source image."""

    #: Recorded beside every mask as its provenance (which backend made it).
    name: ClassVar[str] = "custom"

    @abstractmethod
    def matte(self, image: Image.Image) -> Image.Image:
        """Given an RGBA PIL Image, return a new RGBA Image with the alpha
        channel replaced (or refined) to cut out the background."""
        raise NotImplementedError

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        """The backend's foreground answer alone, as a ``(height, width)``
        uint8 alpha -- the separate mask MASK-1 stores (Ross 2026-09-26: the
        clean copy is never cut out; the client composites this over it).

        This default reads the alpha of ``matte()`` for backends that only
        implement the cut-out; the shipped backends override it so the mask
        is the model's answer without the image's own alpha folded in.
        """
        cutout = self.matte(image.convert("RGBA"))
        return np.asarray(cutout.convert("RGBA"))[..., 3].copy()


class StubMattingBackend(MattingBackend):
    """Default backend: samples a near-uniform backdrop color from the
    image's own corners and thresholds pixel distance from it to transparent
    vs. opaque. This is NOT real subject segmentation -- it's a stand-in
    that produces a real RGBA cutout so the rest of the pipeline (grounding
    scalars, watermark, serve route) can be built and tested against real
    alpha data without a GPU model. Swap via get_matting_backend() /
    Settings.matting_backend once a real backend is deployed.
    """

    #: Euclidean RGB distance from the sampled backdrop color below which a
    #: pixel is considered part of the backdrop (made transparent).
    BACKDROP_DISTANCE_THRESHOLD = 30.0
    #: Corner patch size (pixels) used to sample the backdrop color.
    CORNER_SAMPLE = 5

    name: ClassVar[str] = "stub"

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image.convert("RGBA"))
        h, w = arr.shape[0], arr.shape[1]
        rgb = arr[..., :3].astype(np.float32)

        n = max(1, min(self.CORNER_SAMPLE, h, w))
        corners = np.concatenate(
            [
                rgb[:n, :n].reshape(-1, 3),
                rgb[:n, -n:].reshape(-1, 3),
                rgb[-n:, :n].reshape(-1, 3),
                rgb[-n:, -n:].reshape(-1, 3),
            ]
        )
        backdrop = np.median(corners, axis=0)

        dist = np.sqrt(((rgb - backdrop) ** 2).sum(axis=-1))
        return np.where(dist < self.BACKDROP_DISTANCE_THRESHOLD, 0, 255).astype(np.uint8)

    def matte(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        arr = np.asarray(rgba)
        computed_alpha = self.predict_alpha(rgba)

        # Never make an already-transparent pixel opaque -- only ever cut
        # further into the existing alpha channel.
        existing_alpha = arr[..., 3]
        combined_alpha = np.minimum(computed_alpha, existing_alpha)

        out = np.dstack([arr[..., :3], combined_alpha]).astype(np.uint8)
        return Image.fromarray(out, mode="RGBA")


class BiRefNetMattingBackend(MattingBackend):
    """Real subject-segmentation backend: BiRefNet, run via
    onnxruntime-gpu against ONNX weights baked into the worker Docker
    image (see Dockerfile.worker) at deploy time.

    Model spec (see onnx-community/BiRefNet-ONNX preprocessor_config.json
    on Hugging Face, MIT licensed export of ZhengPeng7/BiRefNet):
      - input: single RGB image, resized to 1024x1024, rescaled to [0, 1],
        then normalized per-channel with ImageNet mean/std, NCHW float32.
      - output: a single-channel (1, 1, 1024, 1024) logit map -- a sigmoid
        must be applied to get a [0, 1] foreground probability (it is NOT
        already a probability). This backend additionally min-max
        rescales the sigmoid output per-image (the contrast-stretch used
        by the reference rembg BiRefNet session), which is a no-op when
        the mask already spans [0, 1] but keeps a saturated/low-contrast
        raw output usable.

    DEPLOY CONCERN, kept out of application code and tests: the
    onnxruntime import and the ONNX session load both happen lazily,
    inside ``_get_session()``, on first inference -- never at module
    import time and never in the constructor. Tests exercise the real
    preprocess/run/postprocess/apply-alpha pipeline by injecting a fake
    session (``session=`` ctor arg) or by overriding ``_run()`` in a
    subclass, so neither onnxruntime nor real model weights are ever
    required outside the deploy image.

    A configured-but-broken backend (missing model path, unreadable
    weights, or a session that loads without the CUDA execution provider
    -- e.g. a CUDA/cuDNN version mismatch against the pin in
    Dockerfile.worker) raises ``RuntimeError`` clearly rather than
    silently degrading to a wrong or CPU-slow result.
    """

    #: BiRefNet's fixed square input resolution.
    INPUT_SIZE = 1024

    name: ClassVar[str] = "birefnet"
    #: ImageNet normalization stats used by the model's own preprocessor.
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        *,
        model_path: str | None = None,
        session: Any | None = None,
    ) -> None:
        """``model_path`` is the deploy-time ONNX weights path
        (Settings.matting_model_path / env MATTING_MODEL_PATH), only
        needed on the real inference path. ``session`` is a test seam: any
        object duck-typing onnxruntime's ``InferenceSession`` (a
        ``get_inputs()`` returning objects with a ``.name``, and a
        ``run(output_names, input_feed)``) can be injected directly,
        bypassing onnxruntime and the model path entirely.
        """
        self._model_path = model_path
        self._session = session

    def _get_session(self) -> Any:
        """Lazily resolve the inference session. Only imports onnxruntime
        and loads weights the first time real inference is needed -- never
        at construction, so building a (possibly unconfigured)
        BiRefNetMattingBackend is always cheap and side-effect free."""
        if self._session is not None:
            return self._session

        if not self._model_path:
            raise RuntimeError(
                "BiRefNetMattingBackend has no ONNX weights configured: "
                "set Settings.matting_model_path (env MATTING_MODEL_PATH) "
                "to the BiRefNet weights baked into the worker image (see "
                "Dockerfile.worker), or pass session= explicitly."
            )

        # Import onnxruntime lazily -- never at module import time -- so
        # application code and tests never need the GPU-only dependency.
        import onnxruntime as ort  # type: ignore[import-not-found]

        try:
            session = ort.InferenceSession(
                self._model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"BiRefNetMattingBackend failed to load ONNX weights from "
                f"{self._model_path!r}: {exc}"
            ) from exc

        if "CUDAExecutionProvider" not in session.get_providers():
            # Never silently degrade to CPU: onnxruntime falls back
            # quietly on a CUDA/cuDNN version mismatch (same hazard class
            # as ultralytics' cu12/cu13 wheel collisions), which would
            # make matting orders of magnitude too slow without any error
            # -- refuse instead so the deploy-time pin gets fixed.
            raise RuntimeError(
                "BiRefNetMattingBackend: onnxruntime session loaded "
                "without CUDAExecutionProvider (silent CPU fallback -- "
                "check the CUDA/cuDNN pin in Dockerfile.worker matches "
                "the installed onnxruntime-gpu build)."
            )

        self._session = session
        return self._session

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """RGBA source -> normalized NCHW float32 model input tensor. Reads
        the RGB channels directly out of the RGBA array (never
        ``.convert("RGB")`` on a PIL Image -- see module docstring)."""
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        rgb_arr = np.ascontiguousarray(np.asarray(rgba)[..., :3])
        resized = Image.fromarray(rgb_arr, mode="RGB").resize(
            (self.INPUT_SIZE, self.INPUT_SIZE), Image.BILINEAR
        )
        arr = np.asarray(resized).astype(np.float32) / 255.0
        mean = np.array(self.MEAN, dtype=np.float32)
        std = np.array(self.STD, dtype=np.float32)
        arr = (arr - mean) / std
        chw = arr.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(chw, axis=0).astype(np.float32)  # NCHW

    def _run(self, input_array: np.ndarray) -> np.ndarray:
        """Run the model. Isolated as its own method so tests can override
        it (subclass) with a fake returning a synthetic logit array of the
        same shape a real onnxruntime session would produce, with no
        onnxruntime import and no real weights involved."""
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_array})
        return np.asarray(outputs[0])  # (1, 1, H, W) raw logits

    def _postprocess(self, raw_output: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Raw (1, 1, H, W) logits -> a uint8 alpha mask resized to the
        source image's own (width, height)."""
        logits = raw_output[:, 0, :, :]
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid: logits -> [0, 1]
        pred = np.squeeze(probs, axis=0)  # (H, W)

        lo, hi = float(pred.min()), float(pred.max())
        if hi > lo:
            pred = (pred - lo) / (hi - lo)

        # round(), not a bare astype(uint8) truncation -- truncating would
        # systematically bias every pixel's alpha down by up to a full
        # level (e.g. a near-saturated 254.99 truncating to 254 instead of
        # the correct 255).
        mask_u8 = np.clip(np.round(pred * 255.0), 0, 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_u8, mode="L").resize(size, Image.BILINEAR)
        return np.asarray(mask_img)

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        raw_output = self._run(self._preprocess(rgba))
        return self._postprocess(raw_output, (rgba.width, rgba.height))

    def matte(self, image: Image.Image) -> Image.Image:
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        arr = np.asarray(rgba)
        computed_alpha = self.predict_alpha(rgba)

        # Never make an already-transparent pixel opaque -- only ever cut
        # further into the existing alpha channel (same rule as
        # StubMattingBackend).
        existing_alpha = arr[..., 3]
        combined_alpha = np.minimum(computed_alpha, existing_alpha)

        out = np.dstack([arr[..., :3], combined_alpha]).astype(np.uint8)
        return Image.fromarray(out, mode="RGBA")


_BACKENDS: dict[str, type[MattingBackend]] = {
    "stub": StubMattingBackend,
    "birefnet": BiRefNetMattingBackend,
}


def get_matting_backend(name: str | None = None) -> MattingBackend:
    """Factory for the configured matting backend. Falls back to the stub
    for an unset or unrecognized name -- the pipeline should never hard-fail
    a whole ingest run because of a matting backend misconfiguration."""
    settings: Settings | None = None
    if name is None:
        from ..config import get_settings

        settings = get_settings()
        name = getattr(settings, "matting_backend", "stub")

    key = (name or "stub").lower()
    if key == "birefnet":
        if settings is None:
            from ..config import get_settings

            settings = get_settings()
        return BiRefNetMattingBackend(model_path=getattr(settings, "matting_model_path", None))

    backend_cls = _BACKENDS.get(key, StubMattingBackend)
    return backend_cls()


def apply_matting(data: bytes, *, backend: MattingBackend | None = None) -> tuple[bytes, str]:
    """Run the matting pipeline on raw source image bytes.

    Detects the real source mime (never trusts a caller assumption), opens
    it, converts to RGBA (never RGB -- see module docstring), runs the
    backend, and re-encodes as PNG (the alpha-capable, lossless format the
    rest of the pipeline expects). Returns ``(rgba_png_bytes, mime)``.
    """
    detect_mime(data)  # sniff the real source mime; never assume one
    img = Image.open(io.BytesIO(data))
    rgba = img.convert("RGBA")

    active_backend = backend or get_matting_backend()
    matted = active_backend.matte(rgba)
    if matted.mode != "RGBA":
        matted = matted.convert("RGBA")

    out = io.BytesIO()
    matted.save(out, format="PNG")
    return out.getvalue(), "image/png"
