import io
import os
import time
import random
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ========= Optional libs for TTS/Conversion =========
try:
    from gtts import gTTS
    GTTS_OK = True
except Exception:
    GTTS_OK = False

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except Exception:
    PYDUB_OK = False

# ========= Config =========
DB_PATH = "records.db"                # On Streamlit Cloud, filesystem is ephemeral; use CSV export to keep data.
WORDBANK_PATH = "wordbank.csv"
AUDIO_CACHE_DIR = Path(".tts_cache")
AUDIO_CACHE_DIR.mkdir(exist_ok=True)

TEACHER_PASSWORD = os.environ.get("TEACHER_PASSWORD", "teacher123")
PLAY_DELAY_SEC = 0.6

# ========= UI =========
st.set_page_config(page_title="Arabic Dictation Tester", page_icon="📝", layout="centered")
st.sidebar.header("Settings")
DEBUG = st.sidebar.checkbox("Debug mode", value=False)
FORCE_WAV = st.sidebar.checkbox("Safari compatibility (force WAV)", value=True)
st.sidebar.caption("If audio fails in Safari, keep this ON.")

# ========= Arabic normalization =========
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

# ========= Data: wordbank =========
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

# ========= SQLite =========
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

# ========= TTS: gTTS -> MP3 -> (optional WAV) =========
def _file_bytes(path): return path.read_bytes() if path.exists() else None
def _save_bytes(path, data): path.write_bytes(data)

def _valid_audio(data: bytes, min_len=2048) -> bool:
    return isinstance(data, (bytes, bytearray)) and len(data) >= min_len

def gtts_make_audio(word: str, force_wav: bool = True):
    """
    Generate audio for `word` using gTTS.
    - Always generates MP3 in-memory.
    - If `force_wav` and pydub/ffmpeg available, converts MP3->WAV (Safari-friendly).
    Returns: (bytes, mime, backend_label)
    """
    if not GTTS_OK:
        return None, None, "gTTS-unavailable"

    # 1) gTTS -> MP3 bytes
    try:
        mp3_buf = io.BytesIO()
        gTTS(text=word, lang="ar").write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        mp3_bytes = mp3_buf.read()
        if not _valid_audio(mp3_bytes):
            return None, None, "gTTS-empty"
    except Exception as e:
        if DEBUG: st.warning(f"gTTS failed: {e}")
        return None, None, "gTTS-error"

    # 2) If forcing WAV and pydub is present, convert
    if force_wav and PYDUB_OK:
        try:
            audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            wav_buf = io.BytesIO()
            audio.export(wav_buf, format="wav")
            wav_buf.seek(0)
            wav_bytes = wav_buf.read()
            if _valid_audio(wav_bytes):
                return wav_bytes, "audio/wav", "gTTS→wav"
        except Exception as e:
            if DEBUG: st.warning(f"MP3→WAV conversion failed: {e}")

    # 3) Fallback to MP3 with correct MIME for Safari
    return mp3_bytes, "audio/mpeg", "gTTS"

def ensure_audio_bytes(word: str):
    """
    Caches final browser-friendly file:
      - If FORCE_WAV: cache/use WAV; else MP3 with audio/mpeg.
      - Use gTTS always on cloud; WAV conversion best-effort.
      - Cache key bumped to v2 to invalidate old MP3s.
    """
    safe = str(abs(hash((word, FORCE_WAV, "v2"))))
    mp3_path = AUDIO_CACHE_DIR / f"{safe}.mp3"
    wav_path = AUDIO_CACHE_DIR / f"{safe}.wav"

    # Serve from cache if valid
    if FORCE_WAV and wav_path.exists():
        b = _file_bytes(wav_path)
        if _valid_audio(b): return b, "audio/wav", "cache:wav"
        else: wav_path.unlink(missing_ok=True)
    if (not FORCE_WAV) and mp3_path.exists():
        b = _file_bytes(mp3_path)
        if _valid_audio(b): return b, "audio/mpeg", "cache:mp3"
        else: mp3_path.unlink(missing_ok=True)

    # Generate with gTTS
    audio_bytes, mime, backend = gtts_make_audio(word, force_wav=FORCE_WAV)

    # Cache
    if audio_bytes and mime:
        if mime == "audio/wav":
            _save_bytes(wav_path, audio_bytes)
        else:
            _save_bytes(mp3_path, audio_bytes)

    return audio_bytes, mime, backend

# ========= App =========
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
                # Let Safari sniff MP3; provide format for WAV
                if mime == "audio/mpeg":
                    st.audio(audio_bytes, start_time=0)  # no format arg for mp3 → better Safari behavior
                else:
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