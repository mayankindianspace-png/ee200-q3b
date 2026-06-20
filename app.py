"""
Q3B: Zapp tain America — Interactive Music Identifier
EE200: Signals, Systems and Networks
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import collections, io, csv, os, tempfile
from pathlib import Path

# ─── Page config ────────────────────────────
st.set_page_config(
    page_title="Zapp tain America 🎵",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-family: 'Courier New', monospace;
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #e07b39, #e74c3c, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #7f8c8d; font-size: 1rem; margin-top: -0.5rem; }
    .result-box {
        background: #1a1a2e; color: #00d2ff;
        border-radius: 8px; padding: 1rem 1.5rem;
        font-size: 1.4rem; font-weight: bold;
        border-left: 4px solid #e07b39;
    }
    .metric-card {
        background: #16213e; border-radius: 8px;
        padding: 0.8rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Core DSP functions ──────────────────────

def load_audio_bytes(uploaded_bytes):
    buf = io.BytesIO(uploaded_bytes)
    try:
        fs, data = wav.read(buf)
    except Exception:
        return None, None
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    return data, int(fs)


def generate_synthetic_song(seed, fs=8000, duration=10.0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    freqs = rng.choice(np.arange(200, 3500, 50), size=6, replace=False)
    audio = sum(0.3 * np.sin(2 * np.pi * f * t) for f in freqs)
    mod = np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t)
    audio *= (0.7 + 0.3 * mod)
    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio) + 1e-8)
    return audio, fs


def compute_spectrogram(audio, fs, nperseg=512):
    noverlap = nperseg * 3 // 4
    f, t, Sxx = signal.spectrogram(
        audio, fs=fs, window='hann',
        nperseg=nperseg, noverlap=noverlap, scaling='spectrum'
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return t, f, Sxx_db


def find_peaks_2d(Sxx_db, neighborhood=10, threshold_db=-35):
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(Sxx_db, size=neighborhood) == Sxx_db
    above = Sxx_db > threshold_db
    peaks = np.argwhere(local_max & above)
    return peaks


def build_hashes(peaks, t, f, fan_out=5, dt_max=2.0):
    hashes = []
    idx = np.argsort(peaks[:, 1])
    peaks = peaks[idx]
    for i, (fi, ti) in enumerate(peaks):
        for j in range(i + 1, min(i + fan_out + 1, len(peaks))):
            fj, tj = peaks[j]
            dt = t[tj] - t[ti]
            if dt > dt_max:
                break
            h = (int(fi), int(fj), int(round(dt * 10)))
            hashes.append((h, t[ti]))
    return hashes


def build_single_peak_hashes(peaks, t, f):
    return [((int(fi),), t[ti]) for (fi, ti) in peaks]


class FingerprintDB:
    def __init__(self):
        self.db = collections.defaultdict(list)
        self.song_list = []

    def index_song(self, name, audio, fs, nperseg=512, use_pairs=True):
        t, f, Sxx_db = compute_spectrogram(audio, fs, nperseg=nperseg)
        peaks = find_peaks_2d(Sxx_db)
        hashes = build_hashes(peaks, t, f) if use_pairs else build_single_peak_hashes(peaks, t, f)
        for h, ta in hashes:
            self.db[h].append((name, ta))
        if name not in self.song_list:
            self.song_list.append(name)

    def identify(self, audio, fs, nperseg=512, use_pairs=True):
        t, f, Sxx_db = compute_spectrogram(audio, fs, nperseg=nperseg)
        peaks = find_peaks_2d(Sxx_db)
        q_hashes = build_hashes(peaks, t, f) if use_pairs else build_single_peak_hashes(peaks, t, f)

        matches = collections.defaultdict(list)
        for h, qt in q_hashes:
            if h in self.db:
                for song, dbt in self.db[h]:
                    matches[song].append(round(dbt - qt, 2))

        scores, histograms = {}, {}
        for song, offsets in matches.items():
            counter = collections.Counter(offsets)
            scores[song] = counter.most_common(1)[0][1]
            histograms[song] = counter

        best = max(scores, key=scores.get) if scores else None
        return best, histograms, scores, peaks, t, f, Sxx_db


# ─── Build/cache the library ────────────────

@st.cache_resource
def get_library_and_db():
    SONGS = [
        ("song_alpha",   0),
        ("song_beta",    1),
        ("song_gamma",   2),
        ("song_delta",   3),
        ("song_epsilon", 4),
        ("song_zeta",    5),
        ("song_eta",     6),
        ("song_theta",   7),
    ]
    library = {}
    for name, seed in SONGS:
        audio, fs = generate_synthetic_song(seed)
        library[name] = (audio, fs)

    db_pairs = FingerprintDB()
    db_singles = FingerprintDB()
    for name, (audio, fs) in library.items():
        db_pairs.index_song(name, audio, fs, use_pairs=True)
        db_singles.index_song(name, audio, fs, use_pairs=False)

    return library, db_pairs, db_singles


# ─── Plot helpers ───────────────────────────

def make_spectrogram_fig(t, f, Sxx_db, peaks, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), facecolor='#0e1117')
    for ax in axes:
        ax.set_facecolor('#0e1117')

    # Spectrogram
    axes[0].pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno', vmin=-80, vmax=0)
    axes[0].set_xlabel("Time (s)", color='white')
    axes[0].set_ylabel("Frequency (Hz)", color='white')
    axes[0].set_title(f"Spectrogram — {title}", color='white', fontsize=10)
    axes[0].tick_params(colors='white')
    for sp in axes[0].spines.values():
        sp.set_color('#333')

    # Constellation
    axes[1].pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno', vmin=-80, vmax=0, alpha=0.6)
    if len(peaks):
        pt = t[peaks[:, 1]]
        pf = f[peaks[:, 0]]
        axes[1].scatter(pt, pf, s=10, c='cyan', marker='o', alpha=0.9)
    axes[1].set_xlabel("Time (s)", color='white')
    axes[1].set_ylabel("Frequency (Hz)", color='white')
    axes[1].set_title(f"Constellation ({len(peaks)} peaks)", color='white', fontsize=10)
    axes[1].tick_params(colors='white')
    for sp in axes[1].spines.values():
        sp.set_color('#333')

    plt.tight_layout()
    return fig


def make_offset_hist_fig(histograms, scores, best):
    top3 = sorted(scores, key=scores.get, reverse=True)[:min(3, len(scores))]
    fig, axes = plt.subplots(1, len(top3), figsize=(5 * len(top3), 3), facecolor='#0e1117')
    if len(top3) == 1:
        axes = [axes]
    for ax, song in zip(axes, top3):
        counter = histograms[song]
        offsets = sorted(counter.keys())
        counts = [counter[o] for o in offsets]
        color = '#00d2ff' if song == best else '#636e72'
        ax.bar(offsets, counts, width=0.1, color=color)
        ax.set_facecolor('#0e1117')
        ax.set_title(f"{'★ ' if song == best else ''}{song}\n(score={scores[song]})",
                     color='white', fontsize=9)
        ax.set_xlabel("Offset (s)", color='white')
        ax.set_ylabel("Matches", color='white')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_color('#333')
    plt.tight_layout()
    return fig


# ─── UI ─────────────────────────────────────

st.markdown('<div class="main-title">🎵 Zapp tain America</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">EE200 · Q3B · Music Fingerprint Identifier</div>', unsafe_allow_html=True)
st.markdown("---")

library, db_pairs, db_singles = get_library_and_db()

tab1, tab2 = st.tabs(["🎧 Single Clip Identifier", "📦 Batch Mode"])

# ════════════════════════════════════════════
# TAB 1: Single clip
# ════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Query Settings")
        query_source = st.radio("Query audio source", ["Synthetic excerpt (demo)", "Upload WAV file"])
        use_pairs = st.toggle("Use paired-hash fingerprint", value=True,
                              help="ON = paired peaks (more robust). OFF = single peaks.")
        nperseg_val = st.select_slider("Spectrogram window size (samples)",
                                       options=[128, 256, 512, 1024, 2048], value=512)

        if query_source == "Synthetic excerpt (demo)":
            query_song = st.selectbox("Select song to query", list(library.keys()))
            excerpt_dur = st.slider("Excerpt duration (seconds)", 1, 10, 5)
            noise_snr = st.slider("Add noise (SNR dB, ∞ = clean)", 0, 60, 60)
            pitch_shift = st.slider("Pitch shift (semitones)", 0, 6, 0)
            run_btn = st.button("🔍 Identify", type="primary", use_container_width=True)
        else:
            uploaded = st.file_uploader("Upload a WAV clip", type=["wav"])
            run_btn = st.button("🔍 Identify", type="primary", use_container_width=True)

    with col_right:
        if run_btn:
            # ── prepare query audio ──
            if query_source == "Synthetic excerpt (demo)":
                q_audio, q_fs = library[query_song]
                q_audio = q_audio[:q_fs * excerpt_dur].copy()
                if noise_snr < 60:
                    sig_pow = np.mean(q_audio ** 2)
                    noise_pow = sig_pow / (10 ** (noise_snr / 10))
                    q_audio = (q_audio + np.random.normal(0, np.sqrt(noise_pow), len(q_audio))).astype(np.float32)
                if pitch_shift > 0:
                    factor = 2 ** (pitch_shift / 12.0)
                    q_audio = signal.resample(q_audio, int(len(q_audio) / factor)).astype(np.float32)
                true_label = query_song
            else:
                if uploaded is None:
                    st.warning("Please upload a WAV file first.")
                    st.stop()
                q_audio, q_fs = load_audio_bytes(uploaded.read())
                if q_audio is None:
                    st.error("Could not read WAV file. Make sure it's a valid PCM WAV.")
                    st.stop()
                true_label = None

            # ── run identifier ──
            db = db_pairs if use_pairs else db_singles
            pred, histograms, scores, peaks, t, f, Sxx_db = db.identify(
                q_audio, q_fs, nperseg=nperseg_val, use_pairs=use_pairs
            )

            # ── result ──
            correct = (pred == true_label) if true_label else None
            res_icon = "✅" if correct else ("❌" if correct is False else "🎵")
            st.markdown(f'<div class="result-box">{res_icon} Matched: <b>{pred or "No match"}</b></div>',
                        unsafe_allow_html=True)
            if true_label:
                st.caption(f"True label: `{true_label}` — {'Correct ✓' if correct else 'Wrong ✗'}")

            st.markdown("#### Spectrogram & Constellation")
            fig1 = make_spectrogram_fig(t, f, Sxx_db, peaks, title=pred or "query")
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

            if histograms:
                st.markdown("#### Offset Histogram (top candidates)")
                fig2 = make_offset_hist_fig(histograms, scores, pred)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            # score table
            if scores:
                st.markdown("#### All candidate scores")
                sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
                cols = st.columns(min(4, len(sorted_scores)))
                for i, (song, sc) in enumerate(sorted_scores[:4]):
                    with cols[i % 4]:
                        st.metric(song, sc, delta=None)

# ════════════════════════════════════════════
# TAB 2: Batch mode
# ════════════════════════════════════════════
with tab2:
    st.subheader("Batch Identification")
    st.markdown("Upload multiple WAV clips. The app will identify each and output `results.csv`.")

    batch_files = st.file_uploader("Upload WAV files", type=["wav"], accept_multiple_files=True)
    use_pairs_batch = st.toggle("Paired hashes (batch)", value=True, key="batch_pairs")

    if st.button("🚀 Run Batch", type="primary") and batch_files:
        rows = []
        progress = st.progress(0)
        status = st.empty()

        for i, f_obj in enumerate(batch_files):
            fname = Path(f_obj.name).stem
            status.text(f"Processing {f_obj.name} …")
            q_audio, q_fs = load_audio_bytes(f_obj.read())
            if q_audio is None:
                pred = "ERROR"
            else:
                db = db_pairs if use_pairs_batch else db_singles
                pred, _, _, _, _, _, _ = db.identify(q_audio, q_fs, use_pairs=use_pairs_batch)
                pred = pred or "unknown"
            rows.append((f_obj.name, pred))
            progress.progress((i + 1) / len(batch_files))

        status.text("Done!")
        progress.progress(1.0)

        # Show table
        st.dataframe({"filename": [r[0] for r in rows],
                      "prediction": [r[1] for r in rows]})

        # CSV download
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["filename", "prediction"])
        writer.writerows(rows)
        st.download_button(
            "⬇ Download results.csv",
            data=csv_buf.getvalue(),
            file_name="results.csv",
            mime="text/csv",
            type="primary"
        )
    elif not batch_files:
        st.info("Upload WAV files above, then click Run Batch.")

# ─── Sidebar: about ─────────────────────────
with st.sidebar:
    st.markdown("### 📚 About")
    st.markdown("""
**EE200 Project — Q3B**

This app implements a Shazam-like music fingerprinter using:
1. **STFT Spectrogram** — time-frequency representation
2. **Constellation** — local peak extraction
3. **Paired-hash fingerprints** — (f₁, f₂, Δt) tuples
4. **Offset histogram** — alignment-based scoring

**Song library**: 8 synthetic songs indexed at startup.

**To use with real audio**: replace the `generate_synthetic_song` calls with actual WAV loads from your song directory.
    """)
    st.markdown("---")
    st.markdown("### 🎛 Indexed songs")
    for s in library.keys():
        st.markdown(f"- `{s}`")
