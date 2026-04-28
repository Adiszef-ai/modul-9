import glob
import json
import os
import uuid
from typing import Optional

import joblib
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

# ── Langfuse setup ───────────────────────────────────────────────────────────
LANGFUSE_ENABLED = False
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    try:
        from langfuse.openai import OpenAI
        from langfuse import get_client as _lf_get_client
        LANGFUSE_ENABLED = True
    except Exception as e:
        from openai import OpenAI
        print(f"[Langfuse] Błąd inicjalizacji: {e} — tracing wyłączony")
else:
    from openai import OpenAI

# ── Konfiguracja ────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
OPENAI_MODEL = "gpt-4o-mini"

LABEL_MAP = {
    "age":            "Wiek",
    "hours_per_week": "Godziny / tydzień",
    "sex":            "Płeć",
    "education":      "Wykształcenie",
    "occupation":     "Zawód",
    "marital_status": "Stan cywilny",
    "workclass":      "Sektor zatrudnienia",
}

# ── MinIO: pobieranie modelu przy starcie kontenera (opcjonalne) ─────────────
def _download_from_minio() -> bool:
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if not endpoint:
        return False
    try:
        import boto3
        from botocore.client import Config

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ["S3_ACCESS_KEY"],
            aws_secret_access_key=os.environ["S3_SECRET_KEY"],
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        bucket = os.environ.get("S3_BUCKET", "adult-income-models")
        os.makedirs(MODELS_DIR, exist_ok=True)
        s3.download_file(bucket, "models/model_latest.pkl",     f"{MODELS_DIR}/model_latest.pkl")
        s3.download_file(bucket, "models/metadata_latest.json", f"{MODELS_DIR}/metadata_latest.json")
        print("[MinIO] ✅ Model pobrany pomyślnie")
        return True
    except Exception as e:
        print(f"[MinIO] ⚠️  Błąd pobierania: {e} — używam lokalnego modelu")
        return False

# ── Ładowanie modelu + metadanych ───────────────────────────────────────────
@st.cache_resource
def load_model_and_meta():
    _download_from_minio()

    # Preferuj _latest (z MinIO), fallback na wersjonowane lokalne
    latest_pkl  = f"{MODELS_DIR}/model_latest.pkl"
    latest_meta = f"{MODELS_DIR}/metadata_latest.json"

    if os.path.exists(latest_pkl):
        pkl_path, meta_path = latest_pkl, latest_meta
    else:
        pkl_files  = sorted(glob.glob(f"{MODELS_DIR}/model_v*.pkl"))
        json_files = sorted(glob.glob(f"{MODELS_DIR}/metadata_v*.json"))
        if not pkl_files:
            st.error("Brak modelu w 'models/'. Uruchom notebook treningowy lub skonfiguruj MinIO.")
            st.stop()
        pkl_path, meta_path = pkl_files[-1], json_files[-1]

    pipeline = joblib.load(pkl_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return pipeline, meta

model, meta = load_model_and_meta()

ALL_FEATURES     = meta["features"]["numeric"] + meta["features"]["categorical"]
NUMERIC_FEATURES = meta["features"]["numeric"]
CAT_FEATURES     = meta["features"]["categorical"]
CAT_VALUES       = meta["categorical_values"]
NUM_RANGES       = meta["numeric_ranges"]

# ── Pydantic schema — structured output dla LLM ─────────────────────────────
class UserFeatures(BaseModel):
    age:            Optional[int] = Field(None, description="Wiek w latach")
    hours_per_week: Optional[int] = Field(None, description="Godziny pracy tygodniowo")
    sex:            Optional[str] = Field(None, description="Płeć")
    education:      Optional[str] = Field(None, description="Poziom wykształcenia")
    occupation:     Optional[str] = Field(None, description="Zawód / stanowisko")
    marital_status: Optional[str] = Field(None, description="Stan cywilny")
    workclass:      Optional[str] = Field(None, description="Sektor zatrudnienia")

# ── Helpers ──────────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    lines = [
        "Jesteś asystentem wyciągającym dane demograficzne i zawodowe z tekstu użytkownika.",
        "Zwróć obiekt JSON z dokładnie tymi polami. Jeśli coś nie wynika z opisu — zwróć null.",
        "",
        "DOZWOLONE WARTOŚCI KATEGORYCZNE (użyj DOKŁADNIE jednej z listy lub null):",
    ]
    for col in CAT_FEATURES:
        vals = ", ".join(f'"{v}"' for v in CAT_VALUES[col])
        lines.append(f"  {col}: [{vals}]")
    lines += [
        "",
        f"  age: int {int(NUM_RANGES['age']['min'])}–{int(NUM_RANGES['age']['max'])}",
        f"  hours_per_week: int {int(NUM_RANGES['hours_per_week']['min'])}–{int(NUM_RANGES['hours_per_week']['max'])}",
        "",
        "Nie wymyślaj wartości. Dopasuj do najbliższej kategorii lub zwróć null.",
    ]
    return "\n".join(lines)


def extract_features(user_text: str, client: OpenAI, trace_id: str | None = None) -> UserFeatures:
    system_prompt = build_system_prompt()

    extra: dict = {}
    if LANGFUSE_ENABLED and trace_id:
        extra["metadata"] = {
            "trace_id":                   trace_id,
            "langfuse_trace_name":        "income-prediction",
            "langfuse_observation_name":  "feature-extraction",
            "langfuse_tags":              ["prediction"],
        }

    resp = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ],
        response_format=UserFeatures,
        temperature=0,
        **extra,
    )
    return resp.choices[0].message.parsed


def do_predict(features: UserFeatures) -> dict:
    data    = features.model_dump()
    missing = [k for k in ALL_FEATURES if data.get(k) is None]
    if missing:
        return {"success": False, "missing": missing}
    X     = pd.DataFrame([{k: data[k] for k in ALL_FEATURES}])
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "success": True,
        "proba":   proba,
        "label":   ">50K USD" if proba >= 0.5 else "≤50K USD",
    }


def log_prediction(trace_id: str | None, result: dict):
    if not (LANGFUSE_ENABLED and trace_id and result.get("success")):
        return
    try:
        _lf_get_client().score(
            trace_id=trace_id,
            name="probability_above_50k",
            value=result["proba"],
            comment=result["label"],
        )
    except Exception as e:
        print(f"[Langfuse] score error: {e}")


def show_prediction(result: dict):
    proba = result["proba"]
    label = result["label"]
    st.divider()
    if proba >= 0.5:
        st.success(f"### Predykcja: **{label}** rocznie")
    else:
        st.info(f"### Predykcja: **{label}** rocznie")
    col_m, col_b = st.columns([1, 2])
    with col_m:
        st.metric("Prawdop. zarobków >50K", f"{proba:.1%}")
    with col_b:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.progress(proba)
    st.caption(
        "⚠️ Predykcja statystyczna na danych Census USA 1994. "
        "Nie stanowi oceny osoby ani gwarancji zarobków."
    )

# ── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ("extracted",    None),
    ("final_result", None),
    ("trace_id",     None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Layout ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predykcja zarobków", page_icon="💰", layout="centered")
st.title("💰 Predykcja zarobków")
st.caption(
    f"Model: LightGBM | "
    f"AUC: {meta['metrics']['cv_auc_mean']:.4f} ± {meta['metrics']['cv_auc_std']:.4f} | "
    f"v{meta['version']}"
)

# Sidebar
with st.sidebar:
    st.header("Konfiguracja")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
    )

    st.divider()

    if LANGFUSE_ENABLED:
        st.success("🔍 Langfuse: aktywny")
    else:
        st.caption("Langfuse: nieaktywny\n(ustaw LANGFUSE_PUBLIC_KEY + SECRET_KEY)")

    if os.environ.get("S3_ENDPOINT_URL"):
        st.success("☁️ MinIO: skonfigurowane")
    else:
        st.caption("MinIO: lokalny model")

    if st.button("🔄 Nowe zapytanie", use_container_width=True):
        for k in ("extracted", "final_result", "trace_id"):
            st.session_state[k] = None
        st.rerun()

    st.divider()
    st.markdown(
        "**Jak działa:**\n"
        "1. Opisujesz sytuację tekstem\n"
        "2. GPT-4o-mini wyciąga dane\n"
        "3. LightGBM przewiduje zarobki\n\n"
        "_Dane: Census USA 1994_"
    )

# Główna sekcja
st.markdown("Opisz swoją sytuację zawodową i demograficzną.")

user_text = st.text_area(
    "Twój opis:",
    placeholder=(
        "Np. 'Mam 38 lat, jestem kobietą, pracuję jako specjalista w prywatnej firmie IT, "
        "mam tytuł magistra, jestem zamężna i pracuję ok. 45 godzin tygodniowo.'"
    ),
    height=130,
)

can_run = bool(user_text.strip()) and bool(api_key.strip())
if not api_key.strip():
    st.warning("Podaj klucz OpenAI API w panelu bocznym.")

if st.button("🔍 Przewiduj zarobki", type="primary", disabled=not can_run):
    client = OpenAI(api_key=api_key)
    trace_id = str(uuid.uuid4())
    st.session_state.trace_id = trace_id

    with st.spinner("Analizuję opis..."):
        try:
            features = extract_features(user_text, client, trace_id=trace_id)
            st.session_state.extracted    = features
            st.session_state.final_result = None
        except Exception as e:
            st.error(f"Błąd LLM: {e}")
            st.stop()

# ── Wyniki ───────────────────────────────────────────────────────────────────
if st.session_state.extracted is not None:
    features: UserFeatures = st.session_state.extracted

    # Podgląd wyciągniętych danych
    with st.expander("📋 Wyciągnięte dane", expanded=True):
        cols = st.columns(2)
        for i, (key, val) in enumerate(features.model_dump().items()):
            label = LABEL_MAP.get(key, key)
            tile  = cols[i % 2]
            if val is None:
                tile.warning(f"**{label}:** ❓ brak")
            else:
                tile.success(f"**{label}:** {val}")

    result = do_predict(features)

    if result["success"]:
        log_prediction(st.session_state.trace_id, result)
        show_prediction(result)
    else:
        # ── Fallback formularz dla brakujących pól ───────────────────────────
        missing       = result["missing"]
        missing_labels = [LABEL_MAP.get(m, m) for m in missing]
        st.warning(
            f"LLM nie wyciągnął: **{', '.join(missing_labels)}**. "
            "Uzupełnij poniżej:"
        )

        with st.form("fallback_form"):
            manual_vals: dict = {}
            for field in missing:
                label = LABEL_MAP.get(field, field)
                if field in CAT_FEATURES:
                    manual_vals[field] = st.selectbox(label, CAT_VALUES[field])
                else:
                    r = NUM_RANGES[field]
                    mid = int((r["min"] + r["max"]) / 2)
                    manual_vals[field] = st.number_input(
                        label,
                        min_value=int(r["min"]),
                        max_value=int(r["max"]),
                        value=mid,
                        step=1,
                    )

            if st.form_submit_button("✅ Przewiduj z uzupełnionymi danymi", type="primary"):
                merged          = {**features.model_dump(), **manual_vals}
                merged_features = UserFeatures(**merged)
                final_result    = do_predict(merged_features)
                if final_result["success"]:
                    log_prediction(st.session_state.trace_id, final_result)
                st.session_state.final_result = final_result

        if st.session_state.final_result and st.session_state.final_result.get("success"):
            show_prediction(st.session_state.final_result)
