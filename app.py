import json
import re
import os
from pathlib import Path
from datetime import timedelta

import boto3
from botocore.exceptions import ClientError

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pycaret.regression import load_model, predict_model

# Pandera validation
import pandera as pa
from pandera import Column, Check, DataFrameSchema


# Optional integrations
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

observe = None
langfuse_client = None

try:
    from langfuse.decorators import observe
    from langfuse import Langfuse

    langfuse_client = Langfuse()
except Exception as e:
    st.error("Langfuse import FAILED")
    st.code(str(e))


# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Half Marathon Predictor", layout="centered")

# global reset handler
AUTO_MODE_LABEL = "Automatyczny (najlepsze dostępne dane)"

if st.session_state.get("btn_reset"):
    st.session_state["user_text"] = ""
    st.session_state.pop("y_hat", None)
    st.session_state["model_mode"] = AUTO_MODE_LABEL
    st.session_state.pop("btn_reset", None)
    st.rerun()

# =========================
# DigitalOcean Spaces config
# =========================
SPACES_BUCKET = os.getenv("SPACES_BUCKET", "amu")
SPACES_ENDPOINT = os.getenv("SPACES_ENDPOINT")
SPACES_KEY = os.getenv("SPACES_KEY")
SPACES_SECRET = os.getenv("SPACES_SECRET")

CACHE_DIR = Path("/tmp/hm_artifacts")

if not all([SPACES_ENDPOINT, SPACES_KEY, SPACES_SECRET]):
    st.error(
        "❌ Brak konfiguracji DigitalOcean Spaces "
        "(SPACES_ENDPOINT / SPACES_KEY / SPACES_SECRET)"
    )
    st.stop()


# -------------------------
# Helpers
# -------------------------

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=SPACES_ENDPOINT,
        aws_access_key_id=SPACES_KEY,
        aws_secret_access_key=SPACES_SECRET,
    )


def load_latest_bundle(model_subdir: str):
    """
    Pobiera artefakty z DO Spaces:
      s3://amu/artifacts/<model_subdir>/
    i zapisuje lokalnie do:
      /tmp/hm_artifacts/<model_subdir>/
    """
    s3 = get_s3_client()

    remote_prefix = f"artifacts/{model_subdir}"
    local_dir = CACHE_DIR / model_subdir
    local_dir.mkdir(parents=True, exist_ok=True)

    # --- latest.json ---
    latest_path = local_dir / "latest.json"

    try:
        s3.download_file(
            SPACES_BUCKET,
            f"{remote_prefix}/latest.json",
            str(latest_path),
        )
    except ClientError as e:
        raise RuntimeError(
            f"Brak latest.json w s3://{SPACES_BUCKET}/{remote_prefix}"
        ) from e


    latest = json.loads(latest_path.read_text(encoding="utf-8"))

    # --- pozostałe pliki ---
    model_pkl = local_dir / latest["model_pkl"]
    meta_json = local_dir / latest["metadata_json"]
    schema_json = local_dir / latest["schema_json"]

    for remote_name, local_path in [
        (latest["model_pkl"], model_pkl),
        (latest["metadata_json"], meta_json),
        (latest["schema_json"], schema_json),
    ]:
        s3.download_file(
            SPACES_BUCKET,
            f"{remote_prefix}/{remote_name}",
            str(local_path),
        )

    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    metadata = json.loads(meta_json.read_text(encoding="utf-8"))

    # PyCaret: ścieżka BEZ .pkl
    model_stem = str(model_pkl.with_suffix(""))
    model = load_model(model_stem)

    return {
        "latest": latest,
        "schema": schema,
        "metadata": metadata,
        "model": model,
        "artifact_dir": local_dir,
        "model_pkl": str(model_pkl),
        "meta_json": str(meta_json),
        "schema_json": str(schema_json),
    }


def lf_flush_safe():
    try:
        langfuse_client.flush()
    except Exception:
        pass


def time_to_seconds(text: str):
    """
    Parses:
      - HH:MM:SS
      - MM:SS
      - also allows "25 min", "25:10", "0:25:10" style
    Returns int seconds or None.
    """
    if text is None:
        return None
    s = str(text).strip().lower()

    # normalize common phrases
    s = s.replace("minut", "min").replace("min.", "min").replace("minutes", "min").replace("minute", "min")
    s = s.replace("sekund", "s").replace("sec.", "s").replace("seconds", "s").replace("second", "s")
    s = s.replace(",", ".").strip()

    # patterns like "25 min 10 s"
    mm = re.search(r"(\d{1,3})\s*min", s)
    ss = re.search(r"(\d{1,2})\s*s", s)
    if mm and ":" not in s:
        m = int(mm.group(1))
        sec = int(ss.group(1)) if ss else 0
        return m * 60 + sec

    # patterns like "1:23:45" or "23:45"
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [int(p) for p in parts]
            if len(parts) == 3:
                h, m, sec = parts
                return h * 3600 + m * 60 + sec
            if len(parts) == 2:
                m, sec = parts
                return m * 60 + sec
        except Exception:
            return None

    # plain integer (assume seconds)
    if re.fullmatch(r"\d+", s):
        return int(s)

    return None


def seconds_to_hhmmss(sec: float):
    if sec is None:
        return ""
    sec = int(round(float(sec)))
    return str(timedelta(seconds=sec))


def normalize_sex(s: str):
    if s is None:
        return None
    x = str(s).strip().upper()
    if x in ["MALE", "MAN", "MĘŻCZYZNA", "MEZCZYZNA", "FACET"]:
        return "M"
    if x in ["FEMALE", "WOMAN", "KOBIETA"]:
        return "K"
    if x == "F":
        return "K"
    if x in ["M", "K"]:
        return x
    return None


@st.cache_resource(ttl=300)
def get_bundles():
    b5 = load_latest_bundle("pre_race_5k")
    try:
        b10 = load_latest_bundle("pre_race_10k")
    except Exception:
        b10 = None
    return b5, b10



def get_default_year_from_schema(schema: dict):
    yrs = schema.get("features", {}).get("Rok", {}).get("allowed", [])
    if isinstance(yrs, list) and len(yrs) > 0:
        try:
            return max(yrs)
        except Exception:
            return 2024
    return 2024


def schema_union_keys(schema_a: dict, schema_b: dict | None):
    keys = set(schema_a.get("features", {}).keys())
    if schema_b:
        keys |= set(schema_b.get("features", {}).keys())
    keys |= {"Wiek", "Płeć", "Czas_5km_sek", "Czas_10km_sek", "Rok"}
    return sorted(keys)


def text_mentions_5k(text: str) -> bool:
    t = (text or "").lower()
    return re.search(r"\b5\s*(km|k)\b", t) is not None


def text_mentions_10k(text: str) -> bool:
    t = (text or "").lower()
    return re.search(r"\b10\s*(km|k)\b", t) is not None

def extract_time_for_distance(text: str, distance_km: int) -> int | None:
    """
    Bezpiecznie wiąże czas z KONKRETNYM dystansem.
    Obsługuje:
      - '5 km 20 min'
      - '20 min na 5 km'
    """
    # 'czas' + 'na' + distance (most common in Poland)
    pat_before = rf"""
        (?P<time>\d{{1,2}}:\d{{2}}(?::\d{{2}})?|\d{{1,3}}\s*min(?:\s*\d{{1,2}}\s*s)?)
        \s*(?:na\s*)?
        {distance_km}\s*(?:km|k)
    """

    # distance + 'czas'
    pat_after = rf"""
        {distance_km}\s*(?:km|k)
        [^0-9k]*?
        (?P<time>\d{{1,2}}:\d{{2}}(?::\d{{2}})?|\d{{1,3}}\s*min(?:\s*\d{{1,2}}\s*s)?)
    """

    for pat in (pat_before, pat_after):
        m = re.search(pat, text, re.VERBOSE)
        if m:
            return time_to_seconds(m.group("time"))

    return None


def regex_fallback_extract(text: str):
    """
    Very simple extraction:
      - age: "mam 35 lat" / "35yo"
      - sex: M/K words
      - 5k time: "5 km 24:10" / "5k 24:10" / "na 5 km 24 min"
      - 10k time: "10 km 50:00" / "10k 50:00"
    """
    t = (text or "").lower()
    out = {}

    # age
    m_age = re.search(r"(\d{1,3})\s*(lat|lata|years|yo|y\.o\.)", t)
    if m_age:
        out["Wiek"] = int(m_age.group(1))

    # sex
    if re.search(r"\b(m|mezczyzna|mężczyzna|facet|male|man)\b", t):
        out["Płeć"] = "M"
    if re.search(r"\b(k|kobieta|female|woman)\b", t):
        out["Płeć"] = "K"

    # 5k time
    out["Czas_5km_sek"] = extract_time_for_distance(t, 5)

    # 10k time
    out["Czas_10km_sek"] = extract_time_for_distance(t, 10)

    return out


def post_normalize_extracted(extracted: dict, user_text: str):
    """
    Twarda ochrona przed 'dopowiadaniem':
    - jeśli tekst NIE wspomina o 5k, a extracted zawiera Czas_5km_sek -> wyzeruj (None)
    - jeśli tekst NIE wspomina o 10k, a extracted zawiera Czas_10km_sek -> wyzeruj (None)
    """
    if "Płeć" in extracted:
        extracted["Płeć"] = normalize_sex(extracted.get("Płeć"))

    # time fields parsing if strings
    for k in ["Czas_5km_sek", "Czas_10km_sek"]:
        if k in extracted and extracted[k] is not None and not isinstance(extracted[k], int):
            extracted[k] = time_to_seconds(str(extracted[k]))

    # age parsing if numeric-like
    if "Wiek" in extracted and extracted["Wiek"] is not None and not isinstance(extracted["Wiek"], int):
        try:
            extracted["Wiek"] = int(float(extracted["Wiek"]))
        except Exception:
            extracted["Wiek"] = None

    # anti-hallucination / anti-inference
    if not text_mentions_5k(user_text):
        extracted["Czas_5km_sek"] = None
    if not text_mentions_10k(user_text):
        extracted["Czas_10km_sek"] = None

    return extracted

def estimate_openai_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-4o-mini",
) -> float:
    PRICES = {
        "gpt-4o-mini": {
            "prompt": 0.15 / 1_000_000,
            "completion": 0.60 / 1_000_000,
        }
    }

    p = PRICES.get(model)
    if not p:
        return 0.0

    return (
        prompt_tokens * p["prompt"]
        + completion_tokens * p["completion"]
    )

def llm_extract_to_dict(text: str, keys: list[str], mode_hint: str):
    """
    Uses OpenAI (if configured) to extract fields into strict JSON.
    Fallback: regex.
    """
    # If no OpenAI client available, fallback
    if OpenAI is None:
        extracted = regex_fallback_extract(text)
        extracted = {k: extracted.get(k, None) for k in keys}
        extracted = post_normalize_extracted(extracted, text)
        return extracted, {"method": "regex", "ok": True, "error": None}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        extracted = regex_fallback_extract(text)
        extracted = {k: extracted.get(k, None) for k in keys}
        extracted = post_normalize_extracted(extracted, text)
        return extracted, {"method": "regex", "ok": True, "error": "OPENAI_API_KEY not set"}

    client = OpenAI()

    system = (
        "You are a strict data extraction engine.\n"
        "Return ONLY valid JSON (no markdown, no commentary).\n"
        "If a field is missing in the user text, set it to null.\n"
        "Time fields must be converted to integer seconds.\n"
        "Sex must be 'M' or 'K'.\n"
        "CRITICAL RULE: Do NOT infer 5k from 10k or 10k from 5k. "
        "Only extract a time if the corresponding distance (5k/5 km or 10k/10 km) is explicitly mentioned.\n"
        "Do not hallucinate values.\n"
    )

    user = f"""
MODE_HINT: {mode_hint}
TARGET_FIELDS: {keys}

User text:
{text}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        
        usage = resp.usage

        cost_usd = estimate_openai_cost_usd(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model="gpt-4o-mini",
        )

        content = resp.choices[0].message.content.strip()
        extracted = json.loads(content)

        # Normalize keys: keep only requested keys
        extracted = {k: extracted.get(k, None) for k in keys}
        extracted = post_normalize_extracted(extracted, text)

        return extracted, {
            "method": "openai",
            "ok": True,
            "error": None,
            "llm": {
                "model": "gpt-4o-mini",
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd_estimated": round(cost_usd, 6),
            },
        }
    
    except Exception as e:
        fallback = regex_fallback_extract(text)
        fallback = {k: fallback.get(k, None) for k in keys}
        fallback = post_normalize_extracted(fallback, text)
        return fallback, {"method": "regex", "ok": False, "error": str(e)}


# Langfuse wrapper via @observe (if available)
if observe is not None:
    llm_extract_to_dict = observe(name="hm_input_extraction")(llm_extract_to_dict)


def prepare_features_for_model(extracted: dict, schema: dict):
    """
    Builds a 1-row DataFrame with exactly the features required by schema.
    Adds default Rok if missing.
    """
    features = schema.get("features", {})
    row = {}

    for k in features.keys():
        row[k] = extracted.get(k, None)

    # default year if missing
    if "Rok" in row and (row["Rok"] is None or row["Rok"] == ""):
        row["Rok"] = get_default_year_from_schema(schema)

    return pd.DataFrame([row])


def build_pandera_schema_from_artifact(schema_json: dict) -> pa.DataFrameSchema:
    """
    nullable=True for all fields (because we enforce ‘required’ separately).
    """
    features = schema_json.get("features", {})
    columns = {}

    for field, rules in features.items():
        t = rules.get("type")

        # default: nullable (we check for required fields separately)
        nullable = True

        if t == "int":
            checks = []
            if "min" in rules:
                checks.append(Check.ge(rules["min"]))
            if "max" in rules:
                checks.append(Check.le(rules["max"]))
            columns[field] = Column(int, checks=checks, nullable=nullable, coerce=True)

        elif t == "category":
            allowed = rules.get("allowed")
            checks = []
            if allowed:
                checks.append(Check.isin(allowed))
            # we leave the categories as object/str
            columns[field] = Column(object, checks=checks, nullable=nullable, coerce=True)

        else:
            # fallback: allow, but do not validate types restrictively
            columns[field] = Column(object, nullable=nullable, coerce=False)

    return DataFrameSchema(columns, coerce=True, strict=False)


def choose_bundle_auto(extracted: dict, b5: dict, b10: dict | None):
    """
    AUTO: select 10K ONLY when:
      - you have 10K artifacts
      - the user has provided 10k
      - the user has provided 5k (because 5k is always required)
    """
    has_5k = extracted.get("Czas_5km_sek") is not None
    has_10k = extracted.get("Czas_10km_sek") is not None

    if b10 and has_10k and has_5k:
        return b10, "PRE_RACE_10K"
    return b5, "PRE_RACE_5K"


def run_prediction(model, features_df: pd.DataFrame) -> float:
    pred = predict_model(model, data=features_df)
    return float(pred["prediction_label"].iloc[0])

def pandera_errors_to_user_messages(
    failure_cases: pd.DataFrame,
    schema: dict
) -> list[str]:
    messages = []

    for _, row in failure_cases.iterrows():
        field = row["column"]
        check = str(row["check"])
        value = row.get("failure_case")

        rules = schema["features"].get(field, {})

        # --- Sex ---
        if field == "Płeć":
            messages.append(
                "❌ **Płeć**: wpisz „M” (mężczyzna) lub „K” (kobieta)."
            )

        # --- Age ---
        elif field == "Wiek":
            min_v = rules.get("min")
            max_v = rules.get("max")
            messages.append(
                f"❌ **Wiek**: podaj liczbę w zakresie **{min_v}–{max_v} lat**."
            )

        # --- Time 5 km ---
        elif field == "Czas_5km_sek":
            min_v = rules.get("min")
            max_v = rules.get("max")
            messages.append(
                "❌ **Czas na 5 km**: wpisz realny czas biegu, np. **5 km biegnę w** "
                "**24:30**, **00:24:30** lub **1470 sekund** "
                f"(zakres: {min_v//60}–{max_v//60} min)."
            )

        # --- Year ---
        elif field == "Rok":
            allowed = rules.get("allowed", [])
            messages.append(
                f"❌ **Rok**: dozwolone wartości to: {', '.join(map(str, allowed))}."
            )

        else:
            messages.append(
                f"❌ **{field}**: nieprawidłowa wartość ({value})."
            )

    # remove duplicates
    return list(dict.fromkeys(messages))

def render_pace_plot(y_hat: float):   
    # --- dystans ---
    total_km = 21
    km = np.arange(1, total_km + 1)

    # --- średnie tempo (min/km) ---
    avg_pace = (y_hat / 60) / 21.0975

    # --- realistyczna zmienność ---
    np.random.seed(42)  # stabilna wizualizacja
    noise = np.random.normal(0, 0.12, size=total_km)      # losowe wahania (~±7 s/km)
    fatigue_trend = np.linspace(0, 0.15, total_km)        # narastające zmęczenie (~+9 s/km)

    # surowa seria tempa
    pace_series = avg_pace + noise + fatigue_trend

    # 🔑 KLUCZOWA POPRAWKA:
    # wymuszamy, aby średnia serii = avg_pace
    pace_series -= pace_series.mean() - avg_pace

    # --- pas zmienności ---
    variability = 0.15  # ~±9 s/km
    lower = pace_series - variability
    upper = pace_series + variability

    # --- wykres ---
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(
        km,
        lower,
        upper,
        alpha=0.25,
        label="Typowa zmienność tempa"
    )

    ax.plot(
        km,
        pace_series,
        linewidth=2,
        label="Szacowane tempo km po km"
    )

    ax.axhline(
        avg_pace,
        linestyle="--",
        linewidth=1.5,
        label="Średnie tempo"
    )

    ax.set_xlabel("Kilometr")
    ax.set_ylabel("Tempo [min/km]")
    ax.set_xlim(1, total_km)
    ax.legend()

    st.pyplot(fig, use_container_width=True) 


# -------------------------
# UI
# -------------------------
st.title("🏃 Predykcja czasu półmaratonu")

b5, b10 = get_bundles()

# Sidebar

with st.sidebar:
    # PREDICTION SETTINGS
    st.header("⚙️ Ustawienia predykcji")

    mode = st.selectbox(
        "Sposób obliczeń",
        [
            "Automatyczny (najlepsze dostępne dane)",
            "Tylko na podstawie czasu na 5 km (5K)",
            "Na podstawie czasu na 5 i 10 km (10K)",
        ],
        index=0,
        key="model_mode",
        help=(
            "W trybie automatycznym aplikacja sama wybierze najlepszy model "
            "w zależności od tego, "
            "jakie dane podasz."
        ),
    )

    st.divider()

    # MODEL ACCURACY
    st.subheader("📊 Dokładność modelu")

    # --- 5 km ---
    mae_5k_sec = b5["metadata"]["metrics"].get(
        "test2024_mae_sec",
        b5["metadata"]["metrics"].get("test_mae_sec", 0),
    )

    mae_5k_min = round(mae_5k_sec / 60, 2)

    st.write("Średni błąd prognozy (model 5K):")
    st.write(f"± {mae_5k_min} min")

    # --- 10 km ---
    if b10:
        mae_10k_sec = b10["metadata"]["metrics"].get(
            "test2024_mae_sec",
            b10["metadata"]["metrics"].get("test_mae_sec", 0),
        )

        mae_10k_min = round(mae_10k_sec / 60, 2)

        st.write("Średni błąd prognozy (model 10K):")
        st.write(f"± {mae_10k_min} min")
        st.caption("Im mniejsza wartość, tym dokładniejsza prognoza.")

    else:
        st.info(
            "Model oparty na czasie 10 km nie jest obecnie dostępny.\n"
            "Predykcja zostanie wykonana na podstawie czasu 5 km."
        )

    st.divider()

    # ADDITIONAL OPTIONS
    st.subheader("🔍 Opcje dodatkowe")

    use_llm = st.checkbox(
        "Wykorzystaj AI do rozpoznania danych z podanego tekstu",
        value=True,
        help=(
            "Wykorzystuje model językowy OpenAI do wyciągnięcia danych z podanego tekstu - w przeciwnym razie REGEX."
        ),
    )

    show_debug = st.checkbox(
        "Pokaż informacje o uzyskanych danych",
        value=True,
        help="Wyświetla dane przekazane do modelu predykcyjnego.",
    )

# Main UI   

user_text = st.text_area(
    label="Wpisz jednym tekstem: wiek, płeć oraz czas na 5 km (w celu uzyskania dokładniejszych szacunków możesz podać również czas na 10 km):",
    height=140,
    placeholder="Np. Cześć, mam 35 lat, jestem mężczyzną, 5 km robię w 24:30, 10 km w 50:10.",
    key="user_text",
)


col1, col2 = st.columns([1, 1])
with col1:
    btn_extract = st.button("🔍 Wyciągnij dane", use_container_width=True)
with col2:
    btn_predict = st.button("🎯 Policz prognozowany czas", use_container_width=True)


if btn_extract or btn_predict:
    try:
        if not user_text.strip():
            st.warning("Wpisz tekst z danymi (płeć, wiek, czas 5 km).")
            st.stop()

        keys = schema_union_keys(b5["schema"], b10["schema"] if b10 else None)

        # Extraction
        if use_llm:
            extracted, meta = llm_extract_to_dict(
                user_text,
                keys,
                mode_hint="pre_race_auto"
            )
        else:
            extracted = regex_fallback_extract(user_text)
            extracted = {k: extracted.get(k, None) for k in keys}
            extracted = post_normalize_extracted(extracted, user_text)
            meta = {"method": "regex", "ok": True, "error": None}

        if btn_predict and extracted.get("Płeć") is None:
            st.error(
                "❌ Nie rozpoznano płci. "
                "Podaj „M” (mężczyzna) lub „K” (kobieta)."
            )
            st.stop()

        mode = st.session_state.model_mode

        if mode == "Na podstawie czasu na 5 i 10 km (10K)":
            if not b10:
                st.error(
                    "Nie masz artefaktów PRE_RACE_10K. "
                    "Wytrenuj model 10K albo wybierz 5K/AUTO."
                )
                st.stop()
            selected_bundle, selected_name = b10, "PRE_RACE_10K"

        elif mode == "Tylko na podstawie czasu na 5 km (5K)":
            selected_bundle, selected_name = b5, "PRE_RACE_5K"

        else:
            # AUTO
            selected_bundle, selected_name = choose_bundle_auto(extracted, b5, b10)

        schema = selected_bundle["schema"]
        metadata = selected_bundle["metadata"]
        model = selected_bundle["model"]

        # Build DF for chosen model schema
        features_df = prepare_features_for_model(extracted, schema)

        # Pandera validation
        p_schema = build_pandera_schema_from_artifact(schema)
        try:
            validated_df = p_schema.validate(features_df, lazy=True)
        except pa.errors.SchemaErrors as e:
            st.error("⚠️ Niektóre dane są nieprawidłowe. Sprawdź poniżej co poprawić:")

            user_messages = pandera_errors_to_user_messages(
                e.failure_cases,
                schema
            )

            for msg in user_messages:
                st.markdown(msg)

            with st.expander("🔍 Szczegóły techniczne (debug)"):
                st.dataframe(e.failure_cases)

            st.stop()

        row = validated_df.iloc[0].to_dict()

        # Consistency 5 km vs. 10 km
        if btn_predict:
            if (
                row.get("Czas_5km_sek") is not None
                and row.get("Czas_10km_sek") is not None
            ):
                pace_5k = row["Czas_5km_sek"] / 5
                pace_10k = row["Czas_10km_sek"] / 10
                delta_pace = pace_10k - pace_5k

                if delta_pace > 20:
                    st.warning(
                        "⚠️ **Niespójne czasy na 5 km i 10 km**\n\n"
                        "Podany czas na 10 km jest znacznie wolniejszy względem 5 km.\n"
                        "Model 10K może dać ostrożniejszą prognozę."
                    )

        # Extraction
        col3, col4 = st.columns([1, 1])

        with col3:
            st.subheader("✅ Ekstrakcja danych")
            st.write("Metoda:", meta["method"], "| OK:", meta["ok"])

            if meta.get("error"):
                st.caption(f"LLM fallback reason: {meta['error']}")

            if show_debug:
                row_for_display = row.copy()
                row_for_display.pop("Rok", None)

                st.code(
                    json.dumps(row_for_display, indent=2, ensure_ascii=False),
                    language="json"
                )

        # Prediction
        with col4:
            if btn_predict:
                y_hat = run_prediction(model, validated_df)
                st.session_state["y_hat"] = y_hat

                MODEL_LABELS = {
                    "PRE_RACE_5K": "model 5K",
                    "PRE_RACE_10K": "model 10K",
                }

                ui_model_name = MODEL_LABELS.get(selected_name, selected_name)

                st.subheader(f"🎯 Wynik – {ui_model_name}")
                st.metric(
                    "Szacowany czas półmaratonu",
                    seconds_to_hhmmss(y_hat)
                )

                mae_sec = metadata.get("metrics", {}).get(
                    "test2024_mae_sec",
                    metadata.get("metrics", {}).get("test_mae_sec"),
                )

                if mae_sec:
                    st.caption(
                        f"Średni błąd (MAE) na teście 2024: "
                        f"±{mae_sec / 60:.2f} min"
                    )

                pace_min_per_km = (y_hat / 60) / 21.0975
                st.write(
                    f"Szacowane tempo: **{pace_min_per_km:.2f} min/km**"
                )

                st.markdown("### 💪 Powodzenia!")
                st.balloons()        

    finally:
        lf_flush_safe()

if "y_hat" in st.session_state:
    y_hat = st.session_state["y_hat"]

    st.divider()
    st.subheader("🏃 Przykładowe tempo biegu – realistyczna wizualizacja")
    render_pace_plot(y_hat)
    st.caption(
        "Wizualizacja przedstawia przykładowy przebieg tempa biegu przy prognozowanym czasie półmaratonu. "
        "Falowanie i pas zmienności ilustrują typowe wahania tempa oraz narastające zmęczenie, "
        "a nie dokładną predykcję tempa na poszczególnych kilometrach."
    )


btn_reset = st.button(
    "🔄 Reset – zacznij od nowa",
    use_container_width=True,
    key="btn_reset",
)