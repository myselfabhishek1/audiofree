"""
streamlit_app.py - Web version of the audio effect, built with Streamlit.

Deploy for FREE on Streamlit Community Cloud (share.streamlit.io):
1. Put this file (as `streamlit_app.py`), `requirements.txt`, and
   `packages.txt` into a public GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", pick the repo, and set the main file to
   `streamlit_app.py`.
4. Deploy.

`packages.txt` installs ffmpeg on the server, which is required for
reading non-WAV formats (mp3, m4a, etc.) via pydub.
"""

import io
import os
import tempfile
from fractions import Fraction

import numpy as np
import requests
import streamlit as st
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly

# ----------------------------- PAYWALL CONFIG -----------------------------
# Paste your Gumroad product's ID here (NOT the permalink - Gumroad requires
# product_id for any product created from Jan 2023 onward).
#
# How to find it: on gumroad.com, edit your product -> "Content" tab ->
# enable "Generate a unique license key per sale" -> the product_id is
# shown right there next to that setting.
GUMROAD_PRODUCT_ID = "XbtF201zjNR6qHclhbH-Gg=="
GUMROAD_PRODUCT_LINK = "https://abhishekreal4.gumroad.com/l/jdcuo"
# ---------------------------------------------------------------------

# ----------------------------- EFFECT CONFIG -----------------------------
LOW_CUTOFF_HZ = 300
HIGH_CUTOFF_HZ = 3000
BLOCK_MS = 20
STAGE1_OVERLAP = 0.6          # slightly higher than 0.5 -> smoother, less robotic
SLICE_PERIOD_MS = 8
SLICE_REMOVE_MS = 1
STAGE2_CROSSFADE_MS = 0.5     # slightly higher than 0.4 -> smoother splices
# ---------------------------------------------------------------------


# ----------------------------- helper functions -----------------------------

def to_float(audio):
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        return audio.astype(np.float64) / max_val
    return audio.astype(np.float64)


def from_float(audio, out_dtype=np.int16):
    """Peak-normalize instead of hard-clipping (hard clipping is a common
    source of harsh/buzzy digital noise)."""
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0.98:
        audio = audio / peak * 0.98
    if np.issubdtype(out_dtype, np.integer):
        max_val = np.iinfo(out_dtype).max
        return (audio * max_val).astype(out_dtype)
    return audio.astype(out_dtype)


def split_bands(signal, sr, low_cutoff, high_cutoff):
    nyq = sr / 2.0
    low_sos = butter(4, low_cutoff / nyq, btype="lowpass", output="sos")
    high_sos = butter(4, high_cutoff / nyq, btype="highpass", output="sos")
    mid_sos = butter(4, [low_cutoff / nyq, high_cutoff / nyq], btype="bandpass", output="sos")
    return (
        sosfiltfilt(low_sos, signal),
        sosfiltfilt(mid_sos, signal),
        sosfiltfilt(high_sos, signal),
    )


def resample_to_length(x, target_len):
    orig_len = len(x)
    if target_len == orig_len or orig_len == 0:
        return x.copy()
    frac = Fraction(target_len, orig_len).limit_denominator(60)
    up, down = frac.numerator, frac.denominator
    y = resample_poly(x, up, down)
    if len(y) != target_len:
        old_idx = np.linspace(0, len(y) - 1, target_len)
        y = np.interp(old_idx, np.arange(len(y)), y)
    return y


def shift_band_frequency_dynamic(signal, sr, block_ms, first_pct, max_abs_pct,
                                  overlap, rng, progress_cb=None):
    """
    Pitch-shifts `signal` using overlap-add, but the shift percentage
    changes every `block_ms` window instead of staying fixed:
      - the first block uses `first_pct`
      - every block after that uses a fresh random value in
        [-max_abs_pct, +max_abs_pct]
    Because this still runs through the same Hann-windowed overlap-add,
    consecutive blocks with different percentages crossfade into each
    other rather than jumping abruptly - this is what keeps the changing
    shift from sounding choppy/robotic.
    """
    n = max(2, int(round(sr * block_ms / 1000.0)))
    hop = max(1, int(round(n * (1 - overlap))))
    window = np.hanning(n)

    num_blocks = max(1, -(-len(signal) // n))  # ceil division
    percents = np.empty(num_blocks)
    percents[0] = first_pct
    if num_blocks > 1:
        percents[1:] = rng.uniform(-max_abs_pct, max_abs_pct, size=num_blocks - 1)

    padded_len = len(signal) + n
    output = np.zeros(padded_len)
    norm = np.zeros(padded_len)

    total_starts = max(1, (len(signal) // hop) + 1)

    for i, start in enumerate(range(0, len(signal), hop)):
        frame = signal[start:start + n]
        if len(frame) < n:
            frame = np.pad(frame, (0, n - len(frame)))

        center = start + n // 2
        block_idx = min(center // n, num_blocks - 1)
        percent = percents[block_idx]
        factor = 1.0 + percent / 100.0

        windowed = frame * window
        shifted_len = max(1, int(round(n / factor)))
        compressed = resample_to_length(windowed, shifted_len)
        restored = resample_to_length(compressed, n)

        output[start:start + n] += restored
        norm[start:start + n] += window

        if progress_cb is not None:
            progress_cb((i + 1) / total_starts)

    norm[norm < 1e-8] = 1.0
    return (output / norm)[:len(signal)]


def apply_frequency_shift(signal, sr, params, rng, progress_cb=None):
    low, mid, high = split_bands(signal, sr, params["low_cutoff"], params["high_cutoff"])

    def sub_progress(base, span):
        if progress_cb is None:
            return None
        return lambda frac: progress_cb(base + span * frac)

    low_s = shift_band_frequency_dynamic(
        low, sr, params["block_ms"], params["low_first_pct"], params["max_random_pct"],
        params["overlap"], rng, sub_progress(0.00, 0.33),
    )
    mid_s = shift_band_frequency_dynamic(
        mid, sr, params["block_ms"], params["mid_first_pct"], params["max_random_pct"],
        params["overlap"], rng, sub_progress(0.33, 0.33),
    )
    high_s = shift_band_frequency_dynamic(
        high, sr, params["block_ms"], params["high_first_pct"], params["max_random_pct"],
        params["overlap"], rng, sub_progress(0.66, 0.34),
    )
    return low_s + mid_s + high_s


def remove_slices(signal, sr, period_ms, remove_ms, crossfade_ms):
    period = int(round(sr * period_ms / 1000.0))
    remove = int(round(sr * remove_ms / 1000.0))
    remove = max(0, min(remove, period - 1))
    fade_n = int(round(sr * crossfade_ms / 1000.0))
    fade_n = max(0, min(fade_n, remove, period // 2))

    if remove <= 0:
        return signal.copy()

    kept_chunks = []
    for start in range(0, len(signal), period):
        chunk = signal[start:start + period]
        if len(chunk) > remove:
            kept_chunks.append(chunk[remove:])

    if not kept_chunks:
        return np.array([], dtype=signal.dtype)
    if fade_n <= 0:
        return np.concatenate(kept_chunks)

    fade_out = np.linspace(1.0, 0.0, fade_n)
    fade_in = np.linspace(0.0, 1.0, fade_n)
    num_chunks = len(kept_chunks)
    lengths = [len(c) for c in kept_chunks]
    positions = [0] * num_chunks
    for i in range(1, num_chunks):
        positions[i] = positions[i - 1] + lengths[i - 1] - fade_n
    total_length = positions[-1] + lengths[-1]
    output = np.zeros(total_length)

    for i, chunk in enumerate(kept_chunks):
        c = chunk.astype(np.float64, copy=True)
        local_fade = min(fade_n, len(c) // 2) if len(c) >= 2 else 0
        if local_fade > 0:
            if i > 0:
                c[:local_fade] *= fade_in[:local_fade]
            if i < num_chunks - 1:
                c[-local_fade:] *= fade_out[-local_fade:]
        start = positions[i]
        output[start:start + len(c)] += c
    return output


def process_channel(channel, sr, params, rng, progress_cb=None):
    def freq_cb(frac):
        if progress_cb is not None:
            progress_cb(0.15 + 0.65 * frac)

    shifted = apply_frequency_shift(channel, sr, params, rng, freq_cb)
    if progress_cb is not None:
        progress_cb(0.85)
    sliced = remove_slices(
        shifted, sr, params["slice_period_ms"], params["slice_remove_ms"], params["crossfade_ms"]
    )
    return sliced


def load_audio_any_format(uploaded_file):
    """Reads WAV/MP3/M4A/OGG/FLAC/etc via pydub+ffmpeg and returns
    (sample_rate, numpy_array, numpy_dtype)."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        seg = AudioSegment.from_file(tmp_path)
    finally:
        os.remove(tmp_path)

    sr = seg.frame_rate
    sample_width = seg.sample_width  # bytes per sample
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    out_dtype = dtype_map.get(sample_width, np.int16)

    samples = np.array(seg.get_array_of_samples())
    if seg.channels == 2:
        samples = samples.reshape((-1, 2))

    return sr, samples, out_dtype


def verify_gumroad_license(product_id, license_key):
    if not license_key.strip():
        return False, "Please enter a license key."
    if product_id == "PASTE_YOUR_PRODUCT_ID_HERE":
    return False, "App owner hasn't configured GUMROAD_PRODUCT_ID yet."
    try:
        response = requests.post(
            "https://api.gumroad.com/v2/licenses/verify",
            data={
                "product_id": product_id,
                "license_key": license_key.strip(),
                "increment_uses_count": "false",
            },
            timeout=10,
        )
        data = response.json()
    except Exception as e:
        return False, f"Could not reach Gumroad: {e}"

    if data.get("success"):
        purchase = data.get("purchase", {})
        if purchase.get("refunded") or purchase.get("chargebacked"):
            return False, "This license's purchase was refunded/reversed."
        return True, "License verified - download unlocked!"
    return False, data.get("message", "Invalid license key.")


# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(page_title="Audio Frequency & Time Glitch Effect")
st.title("Audio Frequency & Time Glitch Effect")
st.markdown(
    "Upload an audio file, adjust the settings, and preview the processed result.\n\n"
    "- **Frequency shift**: low/mid/high bands are pitch-shifted every 20 ms. "
    "The first 20 ms block uses your chosen starting %, then every block after "
    "that gets a fresh random shift (capped at the max % you set).\n"
    "- **Time slicing**: a small piece is removed from every window, crossfaded to avoid clicks."
)

uploaded_file = st.file_uploader(
    "Upload audio",
    type=["wav", "mp3", "m4a", "aac", "ogg", "flac", "wma", "aiff", "opus"],
)

st.subheader("Frequency shift (first 20ms block)")
col1, col2, col3 = st.columns(3)
with col1:
    low_first_pct = st.slider("Low band starting shift (%)", -20.0, 20.0, 2.6, 0.1)
with col2:
    mid_first_pct = st.slider("Mid band starting shift (%)", -20.0, 20.0, 4.3, 0.1)
with col3:
    high_first_pct = st.slider("High band starting shift (%)", -20.0, 20.0, 3.5, 0.1)

max_random_pct = st.slider(
    "Max random shift for every block after the first (%)", 1.0, 20.0, 10.0, 0.5
)

with st.expander("Advanced: time-slicing settings"):
    slice_period_ms = st.slider("Slice window length (ms)", 2, 50, SLICE_PERIOD_MS, 1)
    slice_remove_ms = st.slider("Amount removed per window (ms)", 0.1, 10.0, float(SLICE_REMOVE_MS), 0.1)

if "unlocked" not in st.session_state:
    st.session_state.unlocked = False
if "processed_bytes" not in st.session_state:
    st.session_state.processed_bytes = None

if uploaded_file is not None:
    if st.button("Process Audio", type="primary"):
        progress_bar = st.progress(0, text="Reading file...")

        sr, data, in_dtype = load_audio_any_format(uploaded_file)
        data_f = to_float(data.astype(in_dtype))

        params = dict(
            block_ms=BLOCK_MS,
            low_first_pct=low_first_pct, mid_first_pct=mid_first_pct, high_first_pct=high_first_pct,
            max_random_pct=max_random_pct,
            low_cutoff=LOW_CUTOFF_HZ, high_cutoff=HIGH_CUTOFF_HZ, overlap=STAGE1_OVERLAP,
            slice_period_ms=slice_period_ms, slice_remove_ms=slice_remove_ms,
            crossfade_ms=STAGE2_CROSSFADE_MS,
        )
        rng = np.random.default_rng()

        progress_bar.progress(5, text="Shifting frequencies...")

        if data_f.ndim == 1:
            def cb(frac):
                progress_bar.progress(min(99, 5 + int(85 * frac)), text="Processing...")
            processed = process_channel(data_f, sr, params, rng, cb)
        else:
            channels = []
            n_ch = data_f.shape[1]
            for ch in range(n_ch):
                def cb(frac, ch=ch):
                    overall = (ch + frac) / n_ch
                    progress_bar.progress(min(99, 5 + int(85 * overall)),
                                           text=f"Processing channel {ch + 1}/{n_ch}...")
                channels.append(process_channel(data_f[:, ch], sr, params, rng, cb))
            min_len = min(len(c) for c in channels)
            channels = [c[:min_len] for c in channels]
            processed = np.stack(channels, axis=1)

        progress_bar.progress(95, text="Finalizing...")
        out_audio = from_float(processed, in_dtype)

        buffer = io.BytesIO()
        wavfile.write(buffer, sr, out_audio)
        buffer.seek(0)
        st.session_state.processed_bytes = buffer.getvalue()

        progress_bar.progress(100, text="Done")

    if st.session_state.processed_bytes is not None:
        st.success("Processing complete! Unlock below to download your file.")

        st.divider()
        st.subheader("Unlock download")

        if st.session_state.unlocked:
            st.success("Unlocked! You can download below.")
            st.download_button(
                "Download processed audio",
                data=st.session_state.processed_bytes,
                file_name="processed_output.wav",
                mime="audio/wav",
            )
        else:
            st.markdown("Downloading the full-quality file requires a one-time purchase.")
            st.link_button("Buy access on Gumroad", GUMROAD_PRODUCT_LINK)
            license_key_input = st.text_input("Enter your license key from Gumroad", type="password")
            if st.button("Verify license"):
                is_valid, message = verify_gumroad_license(GUMROAD_PRODUCT_ID, license_key_input)
                if is_valid:
                    st.session_state.unlocked = True
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
else:
    st.info("Upload an audio file above to get started.")
