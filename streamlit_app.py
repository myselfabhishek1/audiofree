"""
streamlit_app.py - Web version of the audio effect, built with Streamlit.

Deploy for FREE on Streamlit Community Cloud (share.streamlit.io):
1. Put this file (renamed to `streamlit_app.py`) plus `requirements.txt`
   into a public GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", pick the repo, and set the main file to
   `streamlit_app.py`.
4. Deploy - you'll get a public URL anyone can open, upload a WAV to,
   and download the processed result from. No payment, ever.

To test locally first (optional):
    pip install streamlit numpy scipy
    streamlit run streamlit_app.py
"""

import io
from fractions import Fraction

import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly

# ----------------------------- helper functions -----------------------------

def to_float(audio):
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        return audio.astype(np.float64) / max_val
    return audio.astype(np.float64)


def from_float(audio, out_dtype=np.int16):
    audio = np.clip(audio, -1.0, 1.0)
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


def shift_band_frequency(signal, sr, block_ms, percent, overlap):
    if percent == 0:
        return signal.copy()
    n = max(2, int(round(sr * block_ms / 1000.0)))
    hop = max(1, int(round(n * (1 - overlap))))
    window = np.hanning(n)
    padded_len = len(signal) + n
    output = np.zeros(padded_len)
    norm = np.zeros(padded_len)
    factor = 1.0 + percent / 100.0

    for start in range(0, len(signal), hop):
        frame = signal[start:start + n]
        if len(frame) < n:
            frame = np.pad(frame, (0, n - len(frame)))
        windowed = frame * window
        shifted_len = max(1, int(round(n / factor)))
        compressed = resample_to_length(windowed, shifted_len)
        restored = resample_to_length(compressed, n)
        output[start:start + n] += restored
        norm[start:start + n] += window

    norm[norm < 1e-8] = 1.0
    return (output / norm)[:len(signal)]


def apply_frequency_shift(signal, sr, block_ms, low_pct, mid_pct, high_pct,
                           low_cutoff, high_cutoff, overlap):
    low, mid, high = split_bands(signal, sr, low_cutoff, high_cutoff)
    low_s = shift_band_frequency(low, sr, block_ms, low_pct, overlap)
    mid_s = shift_band_frequency(mid, sr, block_ms, mid_pct, overlap)
    high_s = shift_band_frequency(high, sr, block_ms, high_pct, overlap)
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


def process_channel(channel, sr, params):
    shifted = apply_frequency_shift(
        channel, sr,
        params["block_ms"], params["low_pct"], params["mid_pct"], params["high_pct"],
        params["low_cutoff"], params["high_cutoff"], params["overlap"],
    )
    sliced = remove_slices(
        shifted, sr, params["slice_period_ms"], params["slice_remove_ms"], params["crossfade_ms"]
    )
    return sliced


# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(page_title="Make audio copyright free")
st.title("Make audio copyright free")
st.markdown(
    "Upload a **WAV** file, adjust the settings, and download the processed result.\n\n"
    "- **Frequency shift**: low/mid/high bands are pitch-shifted every 20 ms.\n"
    "- **Time slicing**: a small piece is removed from every window, crossfaded to avoid clicks."
)

uploaded_file = st.file_uploader("Upload audio (.wav)", type=["wav"])

col1, col2, col3 = st.columns(3)
with col1:
    low_pct = st.slider("Low band shift (%)", -20.0, 20.0, 2.7, 0.1)
with col2:
    mid_pct = st.slider("Mid band shift (%)", -20.0, 20.0, 4.6, 0.1)
with col3:
    high_pct = st.slider("High band shift (%)", -20.0, 20.0, 7.4, 0.1)

col4, col5 = st.columns(2)
with col4:
    slice_period_ms = st.slider("Slice window length (ms)", 2, 50, 20, 1)
with col5:
    slice_remove_ms = st.slider("Amount removed per window (ms)", 0.1, 10.0, 1.0, 0.1)

if uploaded_file is not None:
    if st.button("Process Audio", type="primary"):
        progress_bar = st.progress(0, text="Reading file...")

        sr, data = wavfile.read(io.BytesIO(uploaded_file.read()))
        in_dtype = data.dtype if np.issubdtype(data.dtype, np.integer) else np.int16
        data_f = to_float(data)

        params = dict(
            block_ms=20, low_pct=low_pct, mid_pct=mid_pct, high_pct=high_pct,
            low_cutoff=300, high_cutoff=3000, overlap=0.5,
            slice_period_ms=slice_period_ms, slice_remove_ms=slice_remove_ms,
            crossfade_ms=0.4,
        )

        progress_bar.progress(20, text="Shifting frequencies...")

        if data_f.ndim == 1:
            processed = process_channel(data_f, sr, params)
        else:
            channels = []
            for ch in range(data_f.shape[1]):
                progress_bar.progress(
                    20 + int(60 * ch / data_f.shape[1]),
                    text=f"Processing channel {ch + 1}...",
                )
                channels.append(process_channel(data_f[:, ch], sr, params))
            min_len = min(len(c) for c in channels)
            channels = [c[:min_len] for c in channels]
            processed = np.stack(channels, axis=1)

        progress_bar.progress(90, text="Finalizing...")
        out_audio = from_float(processed, in_dtype)

        buffer = io.BytesIO()
        wavfile.write(buffer, sr, out_audio)
        buffer.seek(0)

        progress_bar.progress(100, text="Done")

        st.audio(buffer, format="audio/wav")
        st.download_button(
            "Download processed audio",
            data=buffer,
            file_name="processed_output.wav",
            mime="audio/wav",
        )
else:
    st.info("Upload a WAV file above to get started.")
