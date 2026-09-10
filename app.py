"""
Arabic Reading Comparison App
==============================
Upload a text file (TXT / PDF / DOCX), read it aloud in Arabic, and get
real-time feedback comparing what you said against what's written.

Deploy for free on Streamlit Community Cloud:
  1. Push app.py + requirements.txt to a public GitHub repo
  2. Go to https://share.streamlit.io -> New app -> pick the repo -> app.py
"""

import base64
import difflib
import io
import struct
import wave

import streamlit as st

# Optional imports handled gracefully (so the app still boots if a
# dependency is briefly missing during first deploy / cold start).
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx  # python-docx
except ImportError:
    docx = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Arabic Reading Comparison",
    page_icon="🎙️",
    layout="wide",
)

if "spoken_text" not in st.session_state:
    st.session_state.spoken_text = ""
if "reference_text" not in st.session_state:
    st.session_state.reference_text = ""
if "trigger_alert" not in st.session_state:
    st.session_state.trigger_alert = False


# --------------------------------------------------------------------------
# File parsing helpers
# --------------------------------------------------------------------------
def extract_text_from_txt(uploaded_file) -> str:
    raw = uploaded_file.read()
    for encoding in ("utf-8", "utf-8-sig", "windows-1256", "cp1256"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="ignore")


def extract_text_from_pdf(uploaded_file) -> str:
    if PdfReader is None:
        st.error("pypdf is not installed. Add 'pypdf' to requirements.txt.")
        return ""
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_docx(uploaded_file) -> str:
    if docx is None:
        st.error("python-docx is not installed. Add 'python-docx' to requirements.txt.")
        return ""
    document = docx.Document(uploaded_file)
    return "\n".join(p.text for p in document.paragraphs)


def extract_text_from_file(uploaded_file) -> str:
    """Dispatch based on file extension."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a .txt, .pdf, or .docx file.")
        return ""


# --------------------------------------------------------------------------
# Speech-to-text (Arabic)
# --------------------------------------------------------------------------
def transcribe_audio_arabic(audio_bytes: bytes) -> str:
    """
    Transcribe recorded audio (WAV) to Arabic text using
    SpeechRecognition's free Google Web Speech API wrapper.

    Note: this free endpoint is rate-limited and intended for light /
    demo use. For production-grade accuracy, swap this function's body
    to call a paid STT provider (e.g. Google Cloud Speech-to-Text,
    Azure Speech, or Whisper) using the same audio_bytes input.
    """
    if sr is None:
        st.error("SpeechRecognition is not installed. Add 'SpeechRecognition' to requirements.txt.")
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="ar-AR")
        return text
    except sr.UnknownValueError:
        st.warning("Could not understand the audio. Please try speaking again more clearly.")
        return ""
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
        return ""
    except Exception as e:  # noqa: BLE001
        st.error(f"Unexpected transcription error: {e}")
        return ""


# --------------------------------------------------------------------------
# Comparison logic
# --------------------------------------------------------------------------
def compare_texts(reference: str, spoken: str):
    """
    Word-level diff between the reference (file) text and the spoken
    (transcribed) text. Returns HTML with mismatches highlighted in red
    and a count of mismatched words.
    """
    ref_words = reference.split()
    spoken_words = spoken.split()

    matcher = difflib.SequenceMatcher(None, ref_words, spoken_words)
    html_parts = []
    mismatch_count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_chunk = ref_words[i1:i2]
        if tag == "equal":
            for w in ref_chunk:
                html_parts.append(f"<span style='color:#1a7f37;'>{w}</span>")
        elif tag == "replace":
            for w in ref_chunk:
                html_parts.append(
                    f"<span style='color:#d1242f; font-weight:bold; "
                    f"background:#ffebe9; border-radius:3px; padding:1px 3px;'>{w}</span>"
                )
                mismatch_count += 1
        elif tag == "delete":
            # Words present in the file but not spoken at all (skipped/omitted)
            for w in ref_chunk:
                html_parts.append(
                    f"<span style='color:#d1242f; text-decoration:underline wavy; "
                    f"background:#ffebe9; border-radius:3px; padding:1px 3px;'>{w}</span>"
                )
                mismatch_count += 1
        elif tag == "insert":
            # Extra words spoken that aren't in the file at this point;
            # not rendered inline (nothing to highlight in the reference),
            # but counted as an error.
            mismatch_count += 1

    html = " ".join(html_parts)
    # Wrap in an RTL container for correct Arabic rendering
    html = (
        "<div dir='rtl' style='font-size:22px; line-height:2.4; "
        "font-family: \"Traditional Arabic\", \"Amiri\", \"Noto Naskh Arabic\", serif;'>"
        f"{html}</div>"
    )
    return html, mismatch_count


# --------------------------------------------------------------------------
# Audio alert (beep) generated on the fly, no external asset needed
# --------------------------------------------------------------------------
def generate_beep_wav_base64(frequency=880, duration_ms=350, volume=0.5, sample_rate=44100) -> str:
    n_samples = int(sample_rate * duration_ms / 1000)
    buffer = io.BytesIO()
    with wave.open(buffer, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            # simple sine wave with a short fade-out to avoid a click
            fade = max(0.0, 1.0 - (i / n_samples))
            sample = int(volume * fade * 32767 * __import__("math").sin(2 * __import__("math").pi * frequency * t))
            wav_file.writeframesraw(struct.pack("<h", sample))
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def play_beep_alert():
    b64_audio = generate_beep_wav_base64()
    st.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🎙️ Arabic Reading Comparison")
st.caption(
    "Upload a text file, read it aloud in Arabic, and see mismatches highlighted in real time."
)

left_col, right_col = st.columns(2)

# ---- Left panel: file upload + reference text ----
with left_col:
    st.subheader("📄 Reference Text")
    uploaded_file = st.file_uploader(
        "Upload a file (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
    )

    if uploaded_file is not None:
        extracted = extract_text_from_file(uploaded_file)
        if extracted:
            st.session_state.reference_text = extracted

    if st.session_state.reference_text:
        st.markdown(
            f"""
            <div dir='rtl' style='font-size:20px; line-height:2.2; max-height:500px;
            overflow-y:auto; padding:12px; border:1px solid #ddd; border-radius:8px;
            font-family: "Traditional Arabic", "Amiri", "Noto Naskh Arabic", serif;'>
            {st.session_state.reference_text.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload a file to see its content here.")

# ---- Right panel: audio input + comparison ----
with right_col:
    st.subheader("🎤 Speak & Compare")

    if sr is None:
        st.error(
            "The 'SpeechRecognition' package is missing, so transcription is disabled. "
            "Add it to requirements.txt and redeploy."
        )

    audio_value = st.audio_input("Record yourself reading the text aloud")

    if audio_value is not None and st.session_state.reference_text:
        with st.spinner("Transcribing..."):
            audio_bytes = audio_value.read()
            transcribed = transcribe_audio_arabic(audio_bytes)

        if transcribed:
            st.session_state.spoken_text = transcribed
            st.success("Transcription complete.")
            st.markdown(f"**You said:** <span dir='rtl'>{transcribed}</span>", unsafe_allow_html=True)

    elif audio_value is not None and not st.session_state.reference_text:
        st.warning("Please upload a reference file first so there's something to compare against.")

    if st.session_state.spoken_text and st.session_state.reference_text:
        st.markdown("---")
        st.subheader("🔍 Comparison Result")
        highlighted_html, mismatch_count = compare_texts(
            st.session_state.reference_text, st.session_state.spoken_text
        )
        st.markdown(highlighted_html, unsafe_allow_html=True)

        if mismatch_count > 0:
            st.error(f"⚠️ {mismatch_count} mismatch(es) detected.")
            play_beep_alert()
        else:
            st.success("✅ Perfect match! No mismatches detected.")

st.markdown("---")
with st.expander("ℹ️ Notes on accuracy & deployment"):
    st.markdown(
        """
- Transcription uses the free Google Web Speech API via `SpeechRecognition`.
  It's fine for demos but is rate-limited — for production use, swap
  `transcribe_audio_arabic()` to call a paid provider (Google Cloud
  Speech-to-Text, Azure Speech, or Whisper) with the same audio bytes.
- `st.audio_input` records short clips per interaction rather than
  continuous streaming audio — this keeps the app simple and works
  reliably on Streamlit Community Cloud, which doesn't support raw
  websocket audio streaming from the browser out of the box.
- Arabic text renders right-to-left (RTL) automatically in both panels.
        """
    )
