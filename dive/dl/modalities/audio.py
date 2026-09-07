"""Audio modality for DIVE Deep Learning - ``dive/dl/modalities/audio.py``.

Audio becomes a fixed-length vector by summarising a log-mel spectrogram over
time: per-band mean and standard deviation, plus a few global statistics. That
representation is what makes a modality-blind MLP viable on sound - a 3-second
clip and a 30-second clip both come out the same width.

Three decoders are tried, in order of what they can read:

1. ``librosa`` - anything ffmpeg/audioread can open, with resampling, and its own
   well-tested mel filterbank.
2. ``soundfile`` - WAV/FLAC/OGG without an external ffmpeg.
3. the standard library's :mod:`wave` module - uncompressed PCM WAV only, with a
   mel filterbank built here in numpy.

Tier 3 is a real fallback, not a pretend one, but it is a *weak baseline*: no
learned front-end, no pretrained audio model, and MP3/M4A files simply cannot be
opened. :attr:`fallback_caveat` states this and ``dive dl doctor`` prints it.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dive.dl.config import DLDataConfig, Modality
from dive.dl.exceptions import DLBackendError, DLDataError
from dive.dl.modalities.base import (
    AUDIO_SUFFIXES,
    ModalityAdapter,
    RawBatch,
    load_media_manifest,
)
from dive.utils.optional import is_available, load_optional

#: Mel bands. 64 gives 128 pooled features - enough to separate speech, music and
#: environmental sound without making the first Linear layer the bottleneck.
DEFAULT_N_MELS = 64

#: Analysis window and hop, in samples at :data:`DEFAULT_SAMPLE_RATE`.
DEFAULT_N_FFT = 1024
DEFAULT_HOP = 512

#: Everything is resampled (or, in tier 3, decimated) to this rate so that clips
#: recorded at different rates produce comparable bands.
DEFAULT_SAMPLE_RATE = 22050

#: Clips are truncated to this many seconds. A 40-minute recording would otherwise
#: hold ~200 MB of float32 in memory for one row of the feature matrix.
DEFAULT_MAX_SECONDS = 30.0

#: Suffixes the stdlib-only tier can decode. Anything else needs librosa/soundfile.
_STDLIB_SUFFIXES = frozenset({".wav"})

#: Global statistics appended after the per-band mean/std and MFCC blocks.
_GLOBAL_FEATURES = (
    "duration_secs",
    "rms",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_contrast",
    "peak_amplitude",
)



def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney-style mel scale, matching librosa's default within rounding."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, fmin: float = 0.0, fmax: Optional[float] = None
) -> np.ndarray:
    """Build a triangular mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``.

    Implemented here rather than imported so the stdlib tier has no dependency
    beyond numpy. Each row is peak-normalised, which keeps the log-magnitudes in a
    comparable range across bands regardless of bandwidth.
    """
    if fmax is None:
        fmax = sample_rate / 2.0
    n_bins = n_fft // 2 + 1
    bin_frequencies = np.linspace(0.0, sample_rate / 2.0, n_bins)

    mel_points = np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    weights = np.zeros((n_mels, n_bins), dtype=np.float64)
    for band in range(n_mels):
        left, centre, right = hz_points[band], hz_points[band + 1], hz_points[band + 2]
        if right <= left:
            continue
        rising = (bin_frequencies - left) / max(centre - left, 1e-9)
        falling = (right - bin_frequencies) / max(right - centre, 1e-9)
        weights[band] = np.clip(np.minimum(rising, falling), 0.0, None)
        peak = weights[band].max()
        if peak > 0:
            weights[band] /= peak
    return weights


class AudioAdapter(ModalityAdapter):
    """Turns audio files into pooled log-mel feature vectors."""

    modality = Modality.AUDIO
    accelerated_packages = ("librosa", "soundfile")
    fallback_caveat = (
        "Without librosa or soundfile, only uncompressed WAV files can be read - "
        "MP3/M4A/OGG clips are skipped - and features come from a numpy log-mel "
        "approximation. This is a weak baseline suitable for coarse separation "
        "(speech vs music, loud vs quiet), not for speech recognition or "
        "fine-grained audio classification."
    )
    bytes_per_sample = 2_800_000.0  # 30 s of float32 mono at 22.05 kHz, plus framing

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(options)
        self._n_mels = int(self.option("n_mels", DEFAULT_N_MELS))
        self._n_fft = int(self.option("n_fft", DEFAULT_N_FFT))
        self._hop = int(self.option("hop_length", DEFAULT_HOP))
        self._sample_rate = int(self.option("sample_rate", DEFAULT_SAMPLE_RATE))
        self._max_seconds = float(self.option("max_seconds", DEFAULT_MAX_SECONDS))
        self._use_mfcc = bool(self.option("use_mfcc", True))
        self._n_mfcc = int(self.option("n_mfcc", 13))
        self._filterbank: Optional[np.ndarray] = None
        self._decoder = "unset"


    # ------------------------------------------------------------------
    def load(self, source: Any, data_config: DLDataConfig) -> RawBatch:
        return load_media_manifest(source, data_config, AUDIO_SUFFIXES, "audio")

    def _paths(self, inputs: Any) -> List[Any]:
        if isinstance(inputs, (str, Path)):
            return [str(inputs)]
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 2:
                return [row for row in inputs]
            if inputs.ndim == 1 and not isinstance(inputs[0], (str, Path)):
                return [inputs]
            if inputs.dtype == object:
                return list(inputs.ravel().tolist())
        if isinstance(inputs, (list, tuple)):
            res = []
            for item in inputs:
                if isinstance(item, np.ndarray):
                    res.append(item)
                else:
                    res.append(str(item))
            return res
        raise DLDataError(
            f"Could not interpret audio input of type {type(inputs).__name__}.",
            "Pass a file path, a list of file paths, or 1-D audio waveform arrays.",
        )

    # ------------------------------------------------------------------
    # decoding
    # ------------------------------------------------------------------
    def _pick_decoder(self) -> str:
        if bool(self.option("use_librosa", True)) and is_available("librosa"):
            return "librosa"
        if bool(self.option("use_soundfile", True)) and is_available("soundfile"):
            return "soundfile"
        return "wave"

    def _decode(self, path: Any) -> Tuple[np.ndarray, int]:
        """Return ``(mono float32 signal in [-1, 1], sample_rate)``."""
        if isinstance(path, np.ndarray):
            return self._decimate(path, self._sample_rate)

        if self._decoder == "librosa":
            librosa = load_optional("librosa", purpose="audio decoding and resampling")
            signal, rate = librosa.load(
                str(path), sr=self._sample_rate, mono=True, duration=self._max_seconds
            )
            return np.asarray(signal, dtype=np.float32), int(rate)

        if self._decoder == "soundfile":
            soundfile = load_optional("soundfile", purpose="audio decoding")
            # Opened rather than read outright so the clip cap can be expressed in
            # frames at the file's own rate instead of guessing a frame count.
            with soundfile.SoundFile(path) as handle:
                rate = int(handle.samplerate)
                wanted = int(max(1, self._max_seconds * rate))
                signal = handle.read(frames=wanted, dtype="float32", always_2d=True)
            mono = np.asarray(signal, dtype=np.float32).mean(axis=1)
            return self._decimate(mono, rate)

        return self._decode_wave(path)

    def _decode_wave(self, path: str) -> Tuple[np.ndarray, int]:
        """Read an uncompressed PCM WAV with the standard library only.

        Channel mixing and integer scaling are done in numpy rather than with
        :mod:`audioop`, which was removed from the standard library in Python 3.13.
        """
        suffix = Path(path).suffix.lower()
        if suffix not in _STDLIB_SUFFIXES:
            raise DLDataError(
                f"'{suffix or path}' cannot be decoded without librosa or soundfile."
            )
        with wave.open(path, "rb") as handle:
            channels = max(1, handle.getnchannels())
            width = handle.getsampwidth()
            rate = handle.getframerate()
            wanted = int(min(handle.getnframes(), max(1, self._max_seconds * rate)))
            raw = handle.readframes(wanted)

        samples = self._pcm_to_float(raw, width, path)
        if channels > 1:
            usable = (samples.size // channels) * channels
            if usable == 0:
                raise DLDataError(f"'{path}' contains no complete audio frames.")
            samples = samples[:usable].reshape(-1, channels).mean(axis=1)
        return self._decimate(samples, int(rate))

    @staticmethod
    def _pcm_to_float(raw: bytes, width: int, path: str) -> np.ndarray:
        """Convert interleaved PCM bytes to float32 in ``[-1, 1]``."""
        if width == 1:
            # 8-bit WAV is unsigned with a 128 midpoint, unlike every wider format.
            return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        if width == 2:
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if width == 3:
            # 24-bit has no numpy dtype: read bytes, then sign-extend into int32 by
            # placing the three bytes in the high-order positions and shifting down.
            usable = (len(raw) // 3) * 3
            triples = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(-1, 3)
            packed = np.zeros((triples.shape[0], 4), dtype=np.uint8)
            packed[:, 1:] = triples
            values = packed.view("<i4").ravel() >> 8
            return values.astype(np.float32) / 8388608.0
        if width == 4:
            return np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        raise DLDataError(f"Unsupported WAV sample width ({width} bytes) in '{path}'.")

    def _decimate(self, signal: np.ndarray, rate: int) -> Tuple[np.ndarray, int]:
        """Resample to the target rate by linear interpolation.

        Not a polyphase resampler - there is no anti-alias filter, so a downsample
        can fold high frequencies into the mel bands. Acceptable because the target
        rate is fixed and the features are broad band energies; librosa's proper
        resampler is used whenever it is installed.
        """
        signal = np.asarray(signal, dtype=np.float32).ravel()
        if rate == self._sample_rate or signal.size == 0:
            return signal, rate
        duration = signal.size / float(rate)
        target_length = max(1, int(duration * self._sample_rate))
        source_positions = np.linspace(0.0, signal.size - 1, num=signal.size, dtype=np.float64)
        target_positions = np.linspace(0.0, signal.size - 1, num=target_length, dtype=np.float64)
        resampled = np.interp(target_positions, source_positions, signal)
        return resampled.astype(np.float32), self._sample_rate

    # ------------------------------------------------------------------
    # featurising
    # ------------------------------------------------------------------
    def width(self) -> int:
        mfcc_width = (2 * self._n_mfcc) if self._use_mfcc else 0
        return 2 * self._n_mels + mfcc_width + len(_GLOBAL_FEATURES)

    def _compute_mfcc(self, mel: np.ndarray) -> np.ndarray:
        """Compute Discrete Cosine Transform (DCT-II) over mel bands for cepstral features."""
        N = mel.shape[0]
        n = np.arange(N)
        k = np.arange(self._n_mfcc)[:, None]
        basis = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        mfccs = (basis @ mel) / np.sqrt(N)
        return np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)]).astype(np.float32)

    def _log_mel(self, signal: np.ndarray, rate: int) -> np.ndarray:
        """Return a ``(n_mels, n_frames)`` log-mel spectrogram."""
        if self._decoder == "librosa":
            librosa = load_optional("librosa", purpose="log-mel spectrogram")
            try:
                mel = librosa.feature.melspectrogram(
                    y=signal,
                    sr=rate,
                    n_fft=self._n_fft,
                    hop_length=self._hop,
                    n_mels=self._n_mels,
                )
                return np.asarray(librosa.power_to_db(mel, ref=np.max), dtype=np.float32)
            except Exception:
                # A librosa version/API mismatch should degrade to the numpy path
                # rather than fail the whole run.
                pass
        return self._numpy_log_mel(signal, rate)

    def _numpy_log_mel(self, signal: np.ndarray, rate: int) -> np.ndarray:
        n_fft, hop = self._n_fft, self._hop
        if signal.size < n_fft:
            signal = np.pad(signal, (0, n_fft - signal.size))
        n_frames = 1 + (signal.size - n_fft) // hop
        window = np.hanning(n_fft).astype(np.float32)

        frames = self._frame(signal, n_fft, hop, n_frames)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2

        if self._filterbank is None or self._filterbank.shape[1] != spectrum.shape[1]:
            self._filterbank = mel_filterbank(self._n_mels, n_fft, rate)
        mel = self._filterbank @ spectrum.T
        return (10.0 * np.log10(np.maximum(mel, 1e-10))).astype(np.float32)

    @staticmethod
    def _frame(signal: np.ndarray, n_fft: int, hop: int, n_frames: int) -> np.ndarray:
        """Cut ``signal`` into overlapping frames of ``n_fft`` samples.

        A strided view rather than a Python loop: a 30 s clip is ~1300 frames, and
        building them one at a time dominates the featurising cost.
        ``sliding_window_view`` arrived in numpy 1.20, so an older numpy falls back
        to the loop instead of failing.
        """
        view = getattr(np.lib.stride_tricks, "sliding_window_view", None)
        if view is not None:
            return view(signal, n_fft)[::hop][:n_frames]
        return np.stack(
            [signal[index * hop : index * hop + n_fft] for index in range(n_frames)]
        )

    def _features(self, path: str) -> np.ndarray:
        signal, rate = self._decode(path)
        if signal.size == 0:
            raise DLDataError(f"'{path}' decoded to an empty signal.")
        mel = self._log_mel(signal, rate)
        band_means = mel.mean(axis=1)
        pooled = np.concatenate([band_means, mel.std(axis=1)])

        blocks: List[np.ndarray] = [pooled]
        if self._use_mfcc:
            blocks.append(self._compute_mfcc(mel))

        duration = signal.size / float(rate or 1)
        rms = float(np.sqrt(np.mean(np.square(signal))))
        crossings = float(np.mean(np.abs(np.diff(np.signbit(signal).astype(np.int8)))))
        # Centre of gravity across mel bands, normalised to [0, 1]. Log-mel values
        # are negative decibels, so they are shifted to be non-negative first -
        # np.average rejects negative weights.
        weights = band_means - band_means.min()
        centroid = (
            float(np.average(np.arange(mel.shape[0]), weights=weights) / max(mel.shape[0] - 1, 1))
            if weights.sum() > 0
            else 0.0
        )
        contrast = float(np.max(band_means) - np.min(band_means))
        peak_amp = float(np.max(np.abs(signal))) if signal.size > 0 else 0.0

        globals_ = np.asarray([duration, rms, crossings, centroid, contrast, peak_amp], dtype=np.float32)
        blocks.append(globals_)
        return np.concatenate(blocks).astype(np.float32)

    def _matrix(self, paths: List[str]) -> np.ndarray:
        width = self.width()
        rows: List[np.ndarray] = []
        first_error: Optional[str] = None
        for position, path in enumerate(paths):
            try:
                rows.append(self._features(path))
            except Exception as exc:
                if first_error is None:
                    name = getattr(path, "shape", None) or Path(str(path)).name
                    first_error = f"{name}: {type(exc).__name__}: {exc}"
                self.failed_indices.append(position)
                rows.append(np.zeros(width, dtype=np.float32))
        if len(self.failed_indices) == len(paths) and paths:
            raise DLBackendError(
                f"None of the {len(paths)} clip(s) could be decoded with the "
                f"'{self._decoder}' decoder. First error - {first_error}",
                "Install a full decoder with 'pip install librosa soundfile', "
                "or run 'dive dl doctor' for guided setup.",
            )
        if self.failed_indices:
            self.notes.append(
                f"{len(self.failed_indices)} of {len(paths)} clip(s) could not be decoded "
                f"and were excluded (first - {first_error})."
            )
        matrix = np.vstack(rows) if rows else np.zeros((0, width), dtype=np.float32)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    def fit_transform(self, inputs: Any) -> np.ndarray:
        paths = self._paths(inputs)
        if not paths:
            raise DLDataError("No audio files to featurise.")
        self.notes = []
        self.failed_indices = []
        self._decoder = self._pick_decoder()

        matrix = self._matrix(paths)
        mfcc_label = f" + {self._n_mfcc} MFCCs" if self._use_mfcc else ""
        self.extractor = (
            f"pooled log-mel ({self._n_mels} bands x mean/std{mfcc_label} + "
            f"{len(_GLOBAL_FEATURES)} global stats, {self._decoder} decoder @ "
            f"{self._sample_rate} Hz)"
        )
        names = (
            [f"mel{index}_mean" for index in range(self._n_mels)]
            + [f"mel{index}_std" for index in range(self._n_mels)]
        )
        if self._use_mfcc:
            names += (
                [f"mfcc{index}_mean" for index in range(self._n_mfcc)]
                + [f"mfcc{index}_std" for index in range(self._n_mfcc)]
            )
        names += list(_GLOBAL_FEATURES)
        self.feature_names = names
        if self._decoder == "wave":
            self.notes.append(
                "Using the standard-library WAV decoder: compressed formats are "
                "unreadable and features are a numpy approximation. " + self.fallback_caveat
            )
        self.fitted = True
        return matrix


    def transform(self, inputs: Any) -> np.ndarray:
        if not self.fitted:
            raise DLDataError("The audio adapter must be fitted before transform().")
        paths = self._paths(inputs)
        self.failed_indices = []
        return self._matrix(paths)

    def coerce(self, data: Any) -> Any:
        return self._paths(data)
