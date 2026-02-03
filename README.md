# 🏃 Half-Marathon Time Predictor

An application for **estimating half-marathon finish time** using data available  
**before race day** (*pre-race inference*).

The project implements a **complete, production-grade ML pipeline**:
- data preparation and validation,
- model training and versioning,
- inference via a Streamlit application,
- structured data extraction from free-form user text (LLM),
- extraction quality monitoring (Langfuse),
- deployment on **Streamlit Community Cloud**,
- independent model versioning via **GitHub Releases**.

---

## 🎯 Project Goal

The goal of this project is to provide a **realistic estimation of half-marathon time**
based on a **minimal set of inputs** that a runner can reasonably know **before the race**.

The project **explicitly avoids data leakage**:
- no use of target-race results,
- no features available only after race start,
- only *pre-race* information is used.

---

## 🧠 Trained Models

The system uses **two complementary predictive models**.

### PRE_RACE_5K

Baseline model, used when only minimal input data is available:
- sex,
- age,
- **5 km race time**.

Model characteristics:
- works with minimal input requirements,
- provides stable predictions,
- achieves a mean absolute error (MAE) of approximately **5 minutes**
  on the 2024 test dataset.

---

### PRE_RACE_10K

Extended model, used when the user additionally provides a **10 km time**:
- sex,
- age,
- 5 km time,
- 10 km time.

Advantages:
- better representation of the runner’s pacing profile,
- lower prediction error compared to the 5K-only model.

The application **automatically selects** this model when the required inputs are present.

---

## 📊 Validation and Interpretability

- Models are validated using a **temporal split**:
  - training: 2023 data,
  - testing: 2024 data.
- This ensures realistic generalization to future race editions.
- Feature importance analysis confirms that:
  - 5 km and 10 km times are the dominant predictors,
  - age acts as a corrective factor,
  - sex and year have marginal influence.

Model behavior aligns well with domain knowledge.

---

## 📦 Model Artifacts

Each model is distributed with a complete set of artifacts:

- **`.pkl` model file** – trained predictive model,
- **`schema.json`** – strict input data contract (types, ranges, allowed values),
- **`metadata.json`** – training context and quality metrics,
- **`latest.json`** – pointer to the currently active model version.

Artifacts are published via **GitHub Releases**, enabling:
- model updates **without redeploying the application**,
- clear and auditable model versioning,
- easy rollback to previous versions,
- future A/B testing scenarios.

The Streamlit app downloads artifacts **dynamically at startup**
and caches them locally (e.g. in `/tmp`).

---

## 🧩 Streamlit Application

The Streamlit application:

- accepts a **single free-text input**,
- uses an **LLM (OpenAI)** to extract structured JSON input,
- falls back to **regex-based extraction** when LLM is unavailable,
- applies **anti-hallucination guards**
  (no distance mentioned → no inferred time),
- validates inputs using **Pandera + `schema.json`**,
- automatically selects the appropriate model (5K / 10K),
- presents results including:
  - predicted half-marathon finish time,
  - estimated average pace (min/km),
  - model error information (MAE),
  - a realistic km-by-km pace visualization.

---

## 🔍 LLM Extraction Monitoring (Langfuse)

LLM-based input extraction is monitored using **Langfuse**:
- trace logging,
- extraction error analysis,
- iterative prompt improvement,
- cost and latency control.

Langfuse integration is **optional** — the application runs fully without it.

---

## ☁️ Deployment Architecture

- **Application code**: GitHub repository
- **Model artifacts**: GitHub Releases
- **Inference frontend**: Streamlit Community Cloud
- **Application deployment**: automatic on push to GitHub
- **Model updates**: new GitHub Release (no app redeploy required)

---

## 🛠️ Technology Stack

- **Python 3.10**
- **PyCaret 3.3.2**
- **scikit-learn**
- **Streamlit**
- **OpenAI SDK (1.x)** 
- **Langfuse** 
- **Pandera**
- **pandas / numpy / scipy**
- **GitHub Releases**
- **Streamlit Community Cloud**

---

## 🚀 Local Development & Execution

To run the application locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Note:**
- Model artifacts are downloaded from GitHub Releases at application startup.
- If OPENAI_API_KEY is not provided, the app automatically falls back to regex-based extraction.

## 📄 Disclaimer

This project is intended for research and engineering purposes only.
Predictions are approximate estimates and should not replace professional coaching or training advice.

---

# 🏃 Half-Marathon Time Predictor

Aplikacja do **szacowania czasu ukończenia półmaratonu** na podstawie danych dostępnych  
**przed startem biegu** (*pre-race inference*).

Projekt obejmuje **pełny, produkcyjny pipeline ML**:
- przygotowanie i walidację danych,
- trenowanie i wersjonowanie modeli,
- inferencję w aplikacji Streamlit,
- ekstrakcję danych wejściowych z tekstu użytkownika (LLM),
- monitoring jakości ekstrakcji (Langfuse),
- wdrożenie na **Streamlit Community Cloud**,
- niezależne wersjonowanie modeli w **GitHub Releases**.

---

## 🎯 Cel projektu

Celem projektu jest **realistyczna estymacja czasu półmaratonu** w oparciu o **minimalny zestaw informacji**, który zawodnik może znać **przed startem biegu**.

Projekt **świadomie unika data leakage**:
- nie używa danych z biegu docelowego,
- nie korzysta z informacji dostępnych dopiero po starcie,
- wykorzystuje wyłącznie cechy znane *pre-race*.

---

## 🧠 Wytrenowane modele

W projekcie zastosowano **dwa komplementarne modele predykcyjne**.

### PRE_RACE_5K

Model bazowy, używany gdy dostępne są tylko podstawowe dane:
- płeć,
- wiek,
- czas uzyskany na dystansie **5 km**.

Cechy modelu:
- działa przy minimalnych wymaganiach wejściowych,
- zapewnia stabilną predykcję,
- osiąga średni błąd bezwzględny (MAE) ≈ **5 minut**  
  na danych testowych z roku 2024.

---

### PRE_RACE_10K

Model rozszerzony, wykorzystywany gdy użytkownik poda dodatkowo czas na **10 km**:
- płeć,
- wiek,
- czas na 5 km,
- czas na 10 km.

Zalety:
- lepsze odwzorowanie tempa zawodnika,
- ok 2 razy niższy błąd predykcji względem wariantu 5 km.

Aplikacja **automatycznie wybiera** ten model, jeśli dane wejściowe są dostępne.

---

## 📊 Walidacja i interpretowalność

- Modele walidowane są **czasowo**:
  - trening: dane z 2023 roku,
  - test: dane z 2024 roku.
- Zapewnia to realistyczną ocenę generalizacji na przyszłe edycje biegu.
- Analiza istotności cech potwierdza, że:
  - kluczową rolę odgrywają czasy na 5 km i 10 km,
  - wiek działa jako korekta,
  - płeć i rok mają wpływ marginalny.

Zachowanie modeli jest zgodne z wiedzą dziedzinową.

---

## 📦 Artefakty modeli

Każdy model posiada kompletny zestaw artefaktów:

- **model `.pkl`** – wytrenowany model predykcyjny,
- **`schema.json`** – kontrakt danych wejściowych (typy, zakresy, dozwolone wartości),
- **`metadata.json`** – metryki jakości i kontekst treningu,
- **`latest.json`** – wskaźnik aktualnej wersji modelu.

Artefakty są publikowane jako **GitHub Releases**, co umożliwia:
- aktualizację modeli **bez redeployu aplikacji**,
- jednoznaczne wersjonowanie modeli,
- prosty rollback do wcześniejszych wersji,
- audyt zmian w czasie.

Aplikacja Streamlit pobiera artefakty **dynamicznie przy starcie**  
i przechowuje je lokalnie w cache (`/tmp`).

---

## 🧩 Aplikacja Streamlit

Aplikacja Streamlit:

- przyjmuje **jedno pole tekstowe** jako wejście,
- wykorzystuje **LLM (OpenAI)** do ekstrakcji danych do postaci JSON,
- posiada **regex fallback**, gdy LLM jest niedostępny,
- stosuje **anti-hallucination guards**  
  (brak wzmianki o dystansie → brak wartości),
- waliduje dane wejściowe przy użyciu **Pandera + `schema.json`**,
- automatycznie dobiera model (5K / 10K),
- prezentuje wynik wraz z:
  - przewidywanym czasem półmaratonu,
  - tempem min/km,
  - informacją o średnim błędzie modelu (MAE),
  - realistyczną wizualizacją tempa km-po-km.

---

## 🔍 Monitoring ekstrakcji LLM (Langfuse)

Ekstrakcja danych wejściowych przez LLM jest monitorowana przy użyciu **Langfuse**:
- logowanie trace’ów,
- analiza błędów ekstrakcji,
- iteracyjne doskonalenie promptów,
- kontrola kosztów i latencji.

Integracja jest **opcjonalna** — aplikacja działa również bez Langfuse.

---

## ☁️ Architektura wdrożeniowa

- **Kod aplikacji**: GitHub
- **Modele i artefakty**: GitHub Releases
- **Frontend / inference**: Streamlit Community Cloud
- **Deploy aplikacji**: automatyczny po pushu do repozytorium
- **Aktualizacja modeli**: publikacja nowego Release (bez redeployu aplikacji)

---

## 🛠️ Stack technologiczny

- **Python 3.10**
- **PyCaret 3.3.2**
- **scikit-learn**
- **Streamlit**
- **OpenAI SDK (1.x)** 
- **Langfuse** 
- **Pandera**
- **pandas / numpy / scipy**
- **GitHub Releases**
- **Streamlit Community Cloud**

---

## 🚀 Uruchomienie lokalne

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Uwaga:**
- Aplikacja pobiera modele z GitHub Releases przy starcie.
- Brak klucza OpenAI powoduje automatyczne przejście na ekstrakcję regex.

## 📄 Disclaimer

Projekt ma charakter badawczo-inżynierski.
Predykcje mają charakter orientacyjny i nie zastępują profesjonalnego planu treningowego.
