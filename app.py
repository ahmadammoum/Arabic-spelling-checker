import io
import os
import time
import random
import sqlite3
import unicodedata
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# =========================
# Optional TTS backends
# =========================
try:
    from gtts import gTTS
    GTTS_OK = True
except Exception:
    GTTS_OK = False

try:
    import pyttsx3
    PYTTSX3_OK = True
except Exception:
    PYTTSX3_OK = False

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except Exception:
    PYDUB_OK = False

# =========================
# Config
# =========================
DB_PATH = "records.db"
WORDBANK_PATH = "wordbank.csv"
AUDIO_CACHE_DIR = Path(".tts_cache")
AUDIO_CACHE_DIR.mkdir(exist_ok=True)
TEACHER_PASSWORD = os.environ.get("TEACHER_PASSWORD", "teacher123")
PLAY_DELAY_SEC = 0.6

# =========================
# UI
# =========================
st.set_page_config(page_title="Arabic Dictation Tester", page_icon="📝", layout="centered")
st.sidebar.header("Settings")
DEBUG = st.sidebar.checkbox("Debug mode", value=False)
FORCE_WAV = st.sidebar.checkbox("Safari compatibility (force WAV)", value=True)
st.sidebar.caption("If audio fails in Safari, keep this ON.")

# =========================
# Arabic normalization
# =========================
ARABIC_DIACRITICS = set([
    "\u0610","\u0611","\u0612","\u0613","\u0614","\u0615","\u0616","\u0617","\u0618","\u0619","\u061A",
    "\u064B","\u064C","\u064D","\u064E","\u064F","\u0650","\u0651","\u0652","\u0653","\u0654","\u0655",
    "\u0656","\u0657","\u0658","\u0659","\u065A","\u065B","\u065C","\u065D","\u065E","\u065F","\u0670"
])

def remove_diacritics(text): return "".join(ch for ch in text if ch not in ARABIC_DIACRITICS)

def normalize_ar(s):
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ـ","").strip()
    s = remove_diacritics(s)
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي")
    return " ".join(s.split())

def is_correct(expected, typed): return normalize_ar(expected) == normalize_ar(typed)

# =========================
# Data: wordbank
# =========================
@st.cache_data
def load_wordbank(path):
    df = pd.read_csv(path)
    if not {"word","difficulty"}.issubset(df.columns):
        raise ValueError("wordbank.csv must have columns: word,difficulty")
    df["word"] = df["word"].astype(str).str.strip()
    df["difficulty"] = df["difficulty"].str.lower().str.strip()
    if not set(df["difficulty"]).issubset({"easy","medium","hard"}):
        raise ValueError("difficulty must be one of: easy, medium, hard")
    if df["word"].duplicated().any():
        raise ValueError("Duplicate words detected in wordbank.")
    return df

def pick_words(df, easy=4, medium=3, hard=3):
    need = {"easy": easy, "medium": medium, "hard": hard}
    selected = []
    for level, n in need.items():
        pool = df[df["difficulty"] == level]["word"].tolist()
        if len(pool) < n:
            raise ValueError(f"Not enough {level} words: need {n}, have {len(pool)}")
        selected.extend(random.sample(pool, n))
    random.shuffle(selected)
    dmap = dict(zip(df["word"], df["difficulty"]))
    return [{"word": w, "difficulty": dmap[w]} for w in selected]

# =========================
# SQLite
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            started_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attempts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            word TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            typed TEXT NOT NULL,
            correct INTEGER NOT NULL,
            answered_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)
    conn.commit(); conn.close()

def create_student(name, cls):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO students(name, class, started_at) VALUES(?,?,?)",
                (name, cls, datetime.now().isoformat(timespec="seconds")))
    sid = cur.lastrowid
    conn.commit(); conn.close()
    return sid

def save_attempt(student_id, word, difficulty, typed, correct):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""INSERT INTO attempts(student_id, word, difficulty, typed, correct, answered_at)
                   VALUES(?,?,?,?,?,?)""",
                (student_id, word, difficulty, typed, int(bool(correct)), datetime.now().isoformat(timespec="seconds")))
    conn.commit(); conn.close()

def fetch_results():
    conn = sqlite3.connect(DB_PATH)
    df_students = pd.read_sql_query("SELECT * FROM students ORDER BY id DESC", conn)
    df_attempts = pd.read_sql_query("SELECT * FROM attempts ORDER BY id DESC", conn)
    conn.close()
    return df_students, df_attempts

# =========================
# Audio helpers (macOS say → AIFF → MP3/WAV)
# =========================
def _file_bytes(path): return path.read_bytes() if path.exists() else None
def _save_bytes(path, data): path.write_bytes(data)

def _run(cmd, env=None):
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except Exception as e:
        if DEBUG: st.warning(f"Command failed: {' '.join(cmd)}\n{e}")
        return False

def _valid_audio(data: bytes, min_len=2048) -> bool:
    # crude sanity check
    return isinstance(data, (bytes, bytearray)) and len(data) >= min_len

def tts_macos_say_to_aiff(path_aiff: Path, text: str, voice: str = "Majed") -> bool:
    """Generate AIFF via macOS 'say' (UTF-8 safe via -f)."""
    if os.uname().sysname.lower() != "darwin":  # macOS only
        return False
    try:
        tmp_txt = AUDIO_CACHE_DIR / f"_say_{abs(hash((text, voice)))}.txt"
        tmp_txt.write_text(text, encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("LC_ALL","en_US.UTF-8"); env.setdefault("LANG","en_US.UTF-8")
        cmd = ["say", "-v", voice, "-f", str(tmp_txt), "-o", str(path_aiff), "--data-format=LEF32@22050"]
        return _run(cmd, env=env) and path_aiff.exists()
    except Exception as e:
        if DEBUG: st.warning(f"say failed: {e}")
        return False

def aiff_to_mp3_bytes(aiff_bytes: bytes) -> bytes | None:
    """AIFF -> MP3 using pydub/ffmpeg."""
    if not PYDUB_OK: return None
    try:
        audio = AudioSegment.from_file(io.BytesIO(aiff_bytes), format="aiff")
        buf = io.BytesIO()
        audio.export(buf, format="mp3")  # needs ffmpeg
        buf.seek(0)
        return buf.read()
    except Exception as e:
        if DEBUG: st.warning(f"pydub MP3 convert failed: {e}")
        return None

def aiff_to_wav_bytes_macos(aiff_path: Path) -> bytes | None:
    """AIFF -> WAV using macOS 'afconvert' (no ffmpeg needed)."""
    try:
        wav_path = aiff_path.with_suffix(".wav")
        cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff_path), str(wav_path)]
        if _run(cmd) and wav_path.exists():
            return wav_path.read_bytes()
        return None
    except Exception as e:
        if DEBUG: st.warning(f"afconvert failed: {e}")
        return None

def tts_gtts_to_mp3_bytes(text: str) -> bytes | None:
    if not GTTS_OK: return None
    try:
        bio = io.BytesIO()
        gTTS(text=text, lang="ar").write_to_fp(bio)
        bio.seek(0)
        return bio.read()
    except Exception as e:
        if DEBUG: st.warning(f"gTTS failed: {e}")
        return None

def tts_pyttsx3_to_wav_bytes(text: str) -> bytes | None:
    if not PYTTSX3_OK: return None
    try:
        tmp_wav = AUDIO_CACHE_DIR / f"_px3_{abs(hash(text))}.wav"
        engine = pyttsx3.init()
        # prefer Arabic-capable voices
        try:
            for v in engine.getProperty("voices"):
                name = (getattr(v,"name","") or "").lower()
                lang = "".join(getattr(v,"languages",[]) or []).lower()
                if "ar" in name or "arab" in name or "ar" in lang or "majed" in name:
                    engine.setProperty("voice", v.id); break
        except Exception: pass
        engine.save_to_file(text, str(tmp_wav)); engine.runAndWait()
        return tmp_wav.read_bytes() if tmp_wav.exists() else None
    except Exception as e:
        if DEBUG: st.warning(f"pyttsx3 failed: {e}")
        return None

def ensure_audio_bytes(word: str):
    """
    Return (audio_bytes, mime, backend_label)

    Priority:
      1) macOS 'say' (Majed) -> AIFF -> WAV (Safari mode) or MP3 (if allowed)
      2) gTTS -> MP3
      3) pyttsx3 -> WAV (then optional MP3)
    Validates audio; regenerates cache if corrupt.
    """
    safe = str(abs(hash(word)))
    mp3_path = AUDIO_CACHE_DIR / f"{safe}.mp3"
    wav_path = AUDIO_CACHE_DIR / f"{safe}.wav"
    aiff_path = AUDIO_CACHE_DIR / f"{safe}.aiff"  # transient

    # 0) Serve from cache if valid
    if FORCE_WAV and wav_path.exists():
        b = _file_bytes(wav_path)
        if _valid_audio(b):
            return b, "audio/wav", "cache:wav"
        else:
            wav_path.unlink(missing_ok=True)

    if (not FORCE_WAV) and mp3_path.exists():
        b = _file_bytes(mp3_path)
        if _valid_audio(b):
            return b, "audio/mp3", "cache:mp3"
        else:
            mp3_path.unlink(missing_ok=True)

    # 1) macOS say -> AIFF
    voice = os.environ.get("MAC_VOICE","Majed")
    if tts_macos_say_to_aiff(aiff_path, word, voice=voice) and aiff_path.exists():
        aiff_bytes = _file_bytes(aiff_path)

        if FORCE_WAV:
            # Prefer WAV (Safari)
            wav = aiff_to_wav_bytes_macos(aiff_path)
            if _valid_audio(wav):
                _save_bytes(wav_path, wav)
                return wav, "audio/wav", f"say→wav({voice})"
            # Fallback to MP3 if conversion to WAV fails and ffmpeg is available
            mp3 = aiff_to_mp3_bytes(aiff_bytes) if aiff_bytes else None
            if _valid_audio(mp3):
                _save_bytes(mp3_path, mp3)
                return mp3, "audio/mp3", f"say→mp3({voice})"
            # Last resort serve AIFF (many browsers won't play)
            if _valid_audio(aiff_bytes):
                return aiff_bytes, "audio/aiff", f"say→aiff({voice})"
        else:
            # Prefer MP3 (if Safari mode off)
            mp3 = aiff_to_mp3_bytes(aiff_bytes) if aiff_bytes else None
            if _valid_audio(mp3):
                _save_bytes(mp3_path, mp3)
                return mp3, "audio/mp3", f"say→mp3({voice})"
            wav = aiff_to_wav_bytes_macos(aiff_path)
            if _valid_audio(wav):
                _save_bytes(wav_path, wav)
                return wav, "audio/wav", f"say→wav({voice})"
            if _valid_audio(aiff_bytes):
                return aiff_bytes, "audio/aiff", f"say→aiff({voice})"

    # 2) gTTS -> MP3
    mp3 = tts_gtts_to_mp3_bytes(word)
    if _valid_audio(mp3):
        if FORCE_WAV and PYDUB_OK:
            # Convert to WAV for Safari, if possible
            try:
                audio = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
                buf = io.BytesIO(); audio.export(buf, format="wav")
                buf.seek(0); wav = buf.read()
                if _valid_audio(wav):
                    _save_bytes(wav_path, wav)
                    return wav, "audio/wav", "gTTS→wav"
            except Exception:
                pass
        _save_bytes(mp3_path, mp3)
        return mp3, "audio/mp3", "gTTS"

    # 3) pyttsx3 -> WAV (then optional MP3)
    wav = tts_pyttsx3_to_wav_bytes(word)
    if _valid_audio(wav):
        if not FORCE_WAV and PYDUB_OK:
            try:
                audio = AudioSegment.from_wav(io.BytesIO(wav))
                buf = io.BytesIO(); audio.export(buf, format="mp3")
                buf.seek(0); data = buf.read()
                if _valid_audio(data):
                    _save_bytes(mp3_path, data)
                    return data, "audio/mp3", "pyttsx3→mp3"
            except Exception:
                pass
        _save_bytes(wav_path, wav)
        return wav, "audio/wav", "pyttsx3→wav"

    return None, None, "none"

# =========================
# App
# =========================
init_db()

mode = st.sidebar.radio("اختر الوضع / Mode", ["👨‍🎓 Student", "👩‍🏫 Teacher Dashboard"])

if mode == "👨‍🎓 Student":
    st.title("📝 اختبار الإملاء العربي")
    st.markdown("<p style='text-align:center; font-weight:bold;'>Done by MR. Ahmad Ammoum</p>", unsafe_allow_html=True)
    st.write("أدخل اسمك وصفّك لبدء الاختبار.")

    with st.form("student_info"):
        name = st.text_input("الاسم الكامل", "")
        cls = st.text_input("الصف (مثال: 9A)", "")
        start_btn = st.form_submit_button("ابدأ الاختبار")

    if start_btn:
        if not name.strip() or not cls.strip():
            st.error("يرجى إدخال الاسم والصف.")
        else:
            st.session_state.student_id = create_student(name.strip(), cls.strip())
            st.session_state.step = 0
            st.session_state.answers = []
            try:
                df_bank = load_wordbank(WORDBANK_PATH)
                st.session_state.quiz = pick_words(df_bank, 4, 3, 3)
                st.success("تم بدء الاختبار.")
            except Exception as e:
                st.error(f"تعذّر تحميل بنك الكلمات: {e}")

    if "student_id" in st.session_state and "quiz" in st.session_state:
        idx = st.session_state.get("step", 0)
        quiz = st.session_state["quiz"]

        if idx < len(quiz):
            item = quiz[idx]
            st.subheader(f"الكلمة رقم {idx+1} من {len(quiz)}")

            audio_bytes, mime, backend = ensure_audio_bytes(item["word"])
            if DEBUG: st.caption(f"TTS backend: {backend}, MIME: {mime}")
            if audio_bytes and mime:
                st.audio(audio_bytes, format=mime, start_time=0)
                time.sleep(PLAY_DELAY_SEC)
            else:
                st.warning("🔈 تعذّر تشغيل الصوت. سيتم عرض الكلمة داخل قسم خاص للمعلم فقط.")
                with st.expander("إظهار/إخفاء الكلمة (للمعلمين)"):
                    st.code(item["word"], language="")

            with st.form(f"answer_{idx}"):
                typed = st.text_input("اكتب الكلمة كما سمعتها:", "")
                submitted = st.form_submit_button("إرسال")

            if submitted:
                corr = is_correct(item["word"], typed)
                save_attempt(st.session_state.student_id, item["word"], item["difficulty"], typed, corr)
                st.session_state.answers.append({
                    "expected": item["word"], "typed": typed,
                    "correct": corr, "difficulty": item["difficulty"]
                })
                st.session_state.step = idx + 1
                st.rerun()

        else:
            results = st.session_state.answers
            total = len(results)
            score = sum(1 for r in results if r["correct"])
            st.success(f"انتهى الاختبار! درجتك: {score} / {total}")

            df_res = pd.DataFrame(results)
            df_res["expected_normalized"] = df_res["expected"].apply(normalize_ar)
            df_res["typed_normalized"] = df_res["typed"].apply(normalize_ar)
            st.dataframe(df_res[["expected","typed","difficulty","correct"]]
                         .rename(columns={"expected":"الكلمة الصحيحة","typed":"إجابتك","difficulty":"المستوى","correct":"صحيح؟"}))
            st.info("يمكنك إغلاق الصفحة الآن. تظهر سجلاتك في لوحة المعلم.")

elif mode == "👩‍🏫 Teacher Dashboard":
    st.title("👩‍🏫 لوحة المعلم")
    st.markdown("<p style='text-align:center; font-weight:bold;'>Done by MR. Ahmad Ammoum</p>", unsafe_allow_html=True)
    pw = st.text_input("كلمة المرور", type="password")
    if pw != TEACHER_PASSWORD:
        st.warning("أدخل كلمة المرور الصحيحة لعرض السجلات.")
        st.stop()

    st.success("تم تسجيل الدخول.")

    df_students, df_attempts = fetch_results()
    if df_students.empty:
        st.info("لا توجد سجلات بعد.")
        st.stop()

    df = df_attempts.merge(df_students, left_on="student_id", right_on="id", suffixes=("_attempt","_student"))
    if df.empty:
        st.info("لا توجد محاولات بعد.")
        st.stop()

    df["correct"] = df["correct"].astype(int)
    with pd.option_context("mode.chained_assignment", None):
        df["answered_at"] = pd.to_datetime(df["answered_at"], errors="coerce")
        df["date"] = df["answered_at"].dt.date

    col1, col2, col3 = st.columns(3)
    with col1: klass = st.text_input("تصفية حسب الصف (اختياري):", "")
    with col2: student_q = st.text_input("تصفية حسب الاسم (اختياري):", "")
    with col3: date_q = st.date_input("تاريخ (اختياري)", value=None)

    df_view = df.copy()
    if klass.strip(): df_view = df_view[df_view["class"].astype(str).str.contains(klass.strip(), case=False, na=False)]
    if student_q.strip(): df_view = df_view[df_view["name"].astype(str).str.contains(student_q.strip(), case=False, na=False)]
    if date_q: df_view = df_view[df_view["date"] == pd.to_datetime(date_q).date()]

    st.subheader("نتائج إجمالية لكل طالب")
    summary = (df_view.groupby(["student_id","name","class"], dropna=False)
                     .agg(total=("correct","count"),
                          score=("correct","sum"),
                          started_at=("started_at","max"))
                     .reset_index().sort_values(["class","name"]))
    summary["percentage"] = (summary["score"]/summary["total"]*100).round(1).astype(str)+"%"
    st.dataframe(summary[["name","class","score","total","percentage","started_at"]])

    st.subheader("محاولات مفصلة")
    detailed = df_view[["name","class","word","difficulty","typed","correct","answered_at"]] \
                .sort_values("answered_at", ascending=False)
    st.dataframe(detailed.rename(columns={
        "name":"الاسم","class":"الصف","word":"الكلمة","difficulty":"المستوى",
        "typed":"إجابة الطالب","correct":"صحيح؟","answered_at":"وقت الإجابة"
    }))

    st.download_button(
        label="تنزيل جميع السجلات (CSV)",
        data=df_view.to_csv(index=False).encode("utf-8-sig"),
        file_name="all_records.csv",
        mime="text/csv"
    )