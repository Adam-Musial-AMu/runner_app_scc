# 🏃 Half-Marathon Time Predictor

Aplikacja do **szacowania czasu ukończenia półmaratonu** na podstawie danych dostępnych  
**przed startem biegu** (*pre-race inference*).

Projekt obejmuje **pełny, produkcyjny pipeline ML**:
- przygotowanie i walidację danych,
- trenowanie i wersjonowanie modeli,
- inferencję w aplikacji Streamlit,
- ekstrakcję danych wejściowych z tekstu użytkownika (LLM),
- monitoring jakości ekstrakcji (Langfuse),
- wdrożenie na **DigitalOcean App Platform**,
- niezależne wersjonowanie modeli w **DigitalOcean Spaces**.

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
- niższy błąd predykcji względem wariantu 5 km.

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

## 📦 Artefakty modelu

Każdy model posiada kompletny zestaw artefaktów:

- **model `.pkl`** – wytrenowany model predykcyjny,
- **`schema.json`** – kontrakt danych wejściowych (typy, zakresy, dozwolone wartości),
- **`metadata.json`** – metryki jakości i kontekst treningu,
- **`latest.json`** – wskaźnik aktualnej wersji modelu używanej przez aplikację.

Artefakty są przechowywane w **DigitalOcean Spaces**, co umożliwia:
- aktualizację modeli **bez redeployu aplikacji**,
- rollback do wcześniejszej wersji,
- przyszłe A/B testy.

---

## 🧩 Aplikacja Streamlit

Aplikacja Streamlit:

- przyjmuje **jedno pole tekstowe** jako wejście,
- wykorzystuje **LLM (OpenAI)** do ekstrakcji danych do postaci JSON,
- posiada **regex fallback**, gdy LLM jest niedostępny,
- stosuje **anti-hallucination guards** (brak wzmianki o dystansie → brak wartości),
- waliduje dane wejściowe przy użyciu **Pandera + `schema.json`**,
- automatycznie dobiera model (5K / 10K),
- prezentuje wynik wraz z:
  - przewidywanym czasem,
  - tempem min/km,
  - informacją o średnim błędzie modelu (MAE).

---

## 🔍 Monitoring LLM (Langfuse)

Ekstrakcja danych wejściowych przez LLM jest monitorowana przy użyciu **Langfuse**:
- logowanie trace’ów,
- analiza błędów ekstrakcji,
- iteracyjne doskonalenie promptów,
- kontrola kosztów i latencji.

---

## ☁️ Architektura wdrożeniowa

- **Kod aplikacji**: GitHub → DigitalOcean App Platform
- **Modele i artefakty**: DigitalOcean Spaces
- **Deploy aplikacji**: automatyczny po pushu do GitHub
- **Aktualizacja modeli**: upload do Spaces (bez deployu)

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
- **DigitalOcean App Platform**
- **DigitalOcean Spaces**

---

## 🚀 Uruchomienie lokalne

```bash
pip install -r requirements.txt
streamlit run app.py

> **Uwaga:**  
> Aplikacja korzysta z modeli przechowywanych w **DigitalOcean Spaces**.  
> Do uruchomienia lokalnego wymagane są odpowiednie zmienne środowiskowe (`SPACES_*`).

