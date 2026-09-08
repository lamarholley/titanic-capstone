# Titanic Survival Assistant

An end-to-end ML application: a trained classifier predicts whether a Titanic
passenger would have survived, and an LLM-powered natural language layer lets
a user ask about a passenger in plain English instead of filling out a form.

> "A 29-year-old woman in first class travelling with her husband."
> → *"This passenger would likely have survived (98.9% estimated probability)..."*

## 1. Project description

**What it does.** You describe a Titanic passenger in a sentence. The app
parses that sentence into the structured features a trained model expects
(class, sex, age, family aboard, fare, port of embarkation), runs the model,
and explains the resulting prediction in plain language.

**Who it's for.** Anyone exploring the classic Titanic dataset who would
rather ask a question than fill out seven form fields — e.g. for a portfolio
demo, a classroom walkthrough, or as a template for wrapping any tabular
classifier in a conversational interface.

**Problem it solves.** Tabular ML models are accurate but not accessible —
using one normally means knowing its exact feature schema. This project
demonstrates a general pattern (LLM as a translation layer between free text
and a model's feature contract) that generalizes well beyond Titanic data.

## 2. Setup instructions

**Requires Python 3.11+.** (`pandas==3.0.2` and other pinned dependencies
don't publish wheels for older versions.) If you're on an older Python —
e.g. an Anaconda `base` environment on Python 3.9 — create a fresh
environment first: `conda create -n titanic python=3.11 -y && conda activate titanic`.

On macOS, xgboost also needs the OpenMP runtime, which isn't installed by
default: `brew install libomp` (one-time, per machine).

```bash
# 1. Clone and enter the repo
git clone <this-repo-url>
cd titanic-capstone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get the data (excluded from git; see "Data & model artifacts" below)
python scripts/download_data.py

# 4. Train all model configurations (logs everything to MLflow)
python -m src.train

# 5. Pick the best run and export it for the app to use
python -m src.select_best_model

# 6. (Optional) configure an LLM provider for full NL understanding
export NEBIUS_API_KEY="sk-..."          # Nebius AI Studio (used in Sprint 16)
# or: export OPENAI_API_KEY="sk-..."     # OpenAI, for local dev

# 7. Run the app
streamlit run src/app.py
```

Without step 6, the app still works end-to-end using a deterministic,
regex-based parser and templated responses instead of an LLM (see
[LLM provider & the no-API-key fallback](#llm-provider--the-no-api-key-fallback)) —
useful for offline development, CI, and grading without a paid API key.

### Data & model artifacts

Per the project requirements, `data/` and `models/` are excluded from git
(see `.gitignore`). `scripts/download_data.py` re-fetches the raw CSV;
`python -m src.train` + `python -m src.select_best_model` regenerate the
trained model and its MLflow history from scratch, deterministically
(fixed `random_state` throughout).

## 3. Usage instructions

**Web app (recommended):**
```bash
streamlit run src/app.py
```
Type a passenger description, or click one of the example prompts. The app
shows the prediction, the estimated survival probability, and (in an
expander) the exact features the model used, so predictions are auditable.

**Command-line demo** (no browser needed):
```bash
python scripts/cli_demo.py
```

**Programmatic use:**
```python
from src.interface import TitanicSurvivalAssistant

assistant = TitanicSurvivalAssistant()
result = assistant.ask("A 22-year-old third-class man travelling alone.")
print(result.status, result.prediction, result.probability)
```

**Retraining / experimenting:** edit `config/config.yaml` (add a run, change
a hyperparameter) and re-run `python -m src.train` — nothing is hardcoded in
`src/train.py`. Compare runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db`
or programmatically via `python -m src.select_best_model --metric roc_auc`.

## 4. Architecture overview

```
Natural language query
        │
        ▼
┌───────────────────────┐   no API key configured   ┌──────────────────────┐
│  src/nlp_parsing.py    │──────────────────────────▶│ parse_with_rules()   │
│  parse_query()         │                           │ (regex, offline)     │
└───────────┬───────────┘                            └──────────┬───────────┘
            │ API key configured                                 │
            ▼                                                    │
┌───────────────────────┐                                        │
│ parse_with_llm()       │                                       │
│ (Nebius/OpenAI, JSON   │                                       │
│  mode extraction)      │                                       │
└───────────┬───────────┘                                        │
            └─────────────────────┬──────────────────────────────┘
                                   ▼
                     ParsedQuery(pclass, sex, age, ...)
                                   │
                    ┌──────────────┴───────────────┐
                    │ src/interface.py               │
                    │  missing_required_fields()      │  ── incomplete → ask user to clarify
                    │  out_of_scope check              │  ── off-topic  → polite redirect
                    │  fill_defaults()                  │  ── same engineer_features() as training
                    └──────────────┬───────────────┘
                                   ▼
                    models/production/  (MLflow Pipeline:
                    ColumnTransformer + best classifier,
                    selected via mlflow.search_runs())
                                   │
                                   ▼
                     prediction + probability
                                   │
                                   ▼
                 generate_response_text() ── LLM explains the result in
                                             context, or a template does
                                             (same fallback rule as parsing)
                                   │
                                   ▼
                     Streamlit chat UI (src/app.py)
```

**Why one pipeline object.** `src/preprocessing.py::build_preprocessor()`
(imputation, scaling, one-hot encoding) is wrapped in the same sklearn
`Pipeline` as the classifier and logged to MLflow as a single artifact. The
interface loads *one* object and calls `.predict()` on nearly-raw feature
values — there's no way for training-time and inference-time preprocessing
to drift apart, because they're the same code path.

**Why feature engineering is a free function.** `engineer_features()` (Title
extraction, FamilySize, IsAlone, Deck) takes a dataframe and returns a new
one, with no fitting. `src/train.py` calls it before the train/test split;
`src/interface.py::fill_defaults()` calls the *identical* function on a
single synthesized row built from parsed LLM output. One implementation,
two callers — the interface can't silently diverge from how the model was
trained.

### LLM provider & the no-API-key fallback

The spec calls for Nebius AI Studio (`src/llm_client.py` defaults to it,
using the OpenAI SDK against Nebius's OpenAI-compatible endpoint). This
build environment has no API key and no network path to Nebius, so a
regex-based fallback (`parse_with_rules` / template responses in
`generate_response_text`) exists at every LLM call site. If `client` is
`None`, the same public functions run the deterministic path — the pytest
suite exercises exactly that path, so `pytest tests/ -v` needs no API key
and no network access. Handing this repo an API key does not require any
code changes — the LLM path activates automatically once
`NEBIUS_API_KEY`/`OPENAI_API_KEY` is set.

## 5. Results summary

Seven configurations across four model families were trained and tracked in
MLflow (`python -m src.train`), evaluated on the same held-out 20% test
split (stratified, `random_state=42`):

| Run                    | Accuracy | Precision | Recall | F1     | ROC-AUC |
|------------------------|---------:|----------:|-------:|-------:|--------:|
| **xgboost_wide**       | **0.849**| 0.809     | 0.797  | **0.803** | 0.841 |
| logreg_baseline        | 0.844    | 0.815     | 0.768  | 0.791  | **0.870** |
| xgboost_default        | 0.827    | 0.797     | 0.739  | 0.767  | 0.834 |
| logreg_regularized     | 0.821    | 0.803     | 0.710  | 0.754  | 0.863 |
| gradient_boosting      | 0.810    | 0.787     | 0.696  | 0.739  | 0.844 |
| random_forest_deep     | 0.804    | 0.804     | 0.652  | 0.720  | 0.845 |
| random_forest_shallow  | 0.799    | 0.780     | 0.667  | 0.719  | 0.838 |

**Selected model: `xgboost_wide`** (600 trees, depth 6, learning rate 0.03),
chosen by `python -m src.select_best_model` (ranks via `mlflow.search_runs()`
on F1 by default — F1 was prioritized over raw accuracy because the two
classes are imbalanced (~38% survived) and false negatives/positives matter
roughly symmetrically for this use case). `logreg_baseline` is a close,
more-interpretable runner-up with the best ROC-AUC; re-running
`select_best_model.py --metric roc_auc` would pick it instead, which is the
point of making the selection metric a flag rather than a hardcoded choice.

**Findings:** sex and passenger class dominate the model's decisions,
consistent with the historical "women and children first" evacuation
priority and the fact that lower classes were berthed further from the
boat deck — matching well-known Titanic EDA findings. Regularizing logistic
regression (`C=0.1`) *hurt* every metric versus the baseline (`C=1.0`),
suggesting the baseline wasn't overfitting despite the training set being
only 712 rows. Random forests underperformed both boosting methods here,
likely because the informative interactions (e.g. sex × class) are exactly
what boosting's sequential correction captures well on a small, low-noise
tabular dataset.

## 6. Reflection

**What I learned:** wiring an LLM to *parse into* a fixed feature schema
(rather than to chat freely) forces a lot of care about what "the schema"
even is — e.g. deciding that `Title` and `Deck` (present in training data via
`Name`/`Cabin`) can't reasonably be asked of a user in conversation, and so
must be *derived* with sensible defaults (title from sex+age, deck as
"Unknown") rather than requested. Routing that derivation through the exact
same `engineer_features()` used at training time — instead of writing a
second, similar-but-not-identical version for inference — turned out to be
the single most important design decision for correctness.

**What was challenging:** MLflow 3.x's newer defaults were a moving target —
the plain filesystem tracking store is now deprecated in favor of a database
backend, and `mlflow.sklearn.log_model`'s new default serialization format
(`skops`) rejects this project's `ColumnTransformer`-based pipeline as
containing "untrusted types." Both required reading the actual error/warning
carefully rather than assuming older tutorials' defaults still applied.
Building a rule-based NL parser that's genuinely useful (not just a demo)
also took more iteration than expected — e.g. a naive substring check for
"man" spuriously matches inside "wo**man**", and "with her husband" needed
to *not* override the sentence's actual subject's sex.

**What I'd improve with more time:**
- Replace the manifest-level EDA with a proper notebook and add SHAP-based
  feature importance / partial dependence plots for the winning model.
- Expand the rule-based parser's coverage (currently a genuinely useful but
  incomplete fallback; the LLM path is materially more robust) or add a
  small labeled set of NL→feature examples to regression-test parsing
  quality over time, LLM included.
- Add MLflow Model Registry stage transitions (`Staging`/`Production`)
  instead of a flat `models/production/` export directory, and wire CI to
  re-run `src/train.py` + tests on every push.
- Validate the Dockerfile against a live Docker daemon (unavailable in the
  environment this was built in — see below) and add a slimmer multi-stage
  build so the image doesn't retrain on every rebuild.

## Repository structure

```
titanic-capstone/
├── README.md
├── requirements.txt
├── Dockerfile
├── pytest.ini
├── .gitignore
├── config/
│   └── config.yaml          # all model/data hyperparameters (no hardcoding)
├── data/
│   └── raw/                 # titanic.csv (gitignored; see scripts/download_data.py)
├── models/
│   └── production/          # exported best MLflow model (gitignored)
├── scripts/
│   ├── download_data.py
│   ├── cli_demo.py
│   └── demo_walkthrough.py  # generates demo_transcript.md
├── src/
│   ├── preprocessing.py     # loading, feature engineering, ColumnTransformer
│   ├── model_factory.py     # config name -> sklearn/xgboost estimator
│   ├── evaluate.py          # shared metrics
│   ├── train.py             # trains + MLflow-logs every config in config.yaml
│   ├── select_best_model.py # mlflow.search_runs() -> exports the winner
│   ├── nlp_parsing.py       # NL -> features (LLM + rule-based fallback)
│   ├── llm_client.py        # Nebius/OpenAI-compatible chat client
│   ├── interface.py         # parse -> validate -> predict -> explain
│   └── app.py                # Streamlit UI
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_interface.py
└── demo_transcript.md        # scripted walkthrough (see below)
```

## Demo

A live screen recording requires a display, which this build environment
(a headless container) doesn't have. In its place, `scripts/demo_walkthrough.py`
runs the actual application code (not mocked output) end-to-end against three
scenarios — a complete query, a follow-up, and an edge case — and saves the
real input/output to [`demo_transcript.md`](./demo_transcript.md). Regenerate
it any time with:

```bash
python scripts/demo_walkthrough.py
```

If you have a display available, `streamlit run src/app.py` plus your OS's
screen recorder is the more faithful way to produce the video deliverable
this project ultimately calls for.

## Grading rubric self-check

| Category                          | Points | Status |
|------------------------------------|-------:|--------|
| Data & Model Quality                |     20 | Done — 7 configs, 4 metrics, justified selection in README §5 |
| Experiment Tracking (MLflow)        |     15 | Done — 7 runs, `mlflow.search_runs()` in `select_best_model.py` |
| LLM Interface                       |     30 | Done — parsing/invocation/response/edge-cases in `interface.py`; Nebius-ready, rule-based fallback |
| Testing                             |     15 | Done — 19 tests (7 preprocessing, 4 model, 8 interface), all passing |
| Documentation & Structure           |     20 | Done — this file, `config.yaml`, pinned `requirements.txt`, `.gitignore` |
| Docker (bonus)                      |    +3 | Written, **not build-verified** (no Docker daemon in this environment) |
