# Chef Mystico — a hard-mode food guessing game

A full-stack machine-learning game. A curated catalogue of 139 regional dishes — the
kind a serious food enthusiast has to think about — behind three play modes, a
PyTorch classifier, and a FastAPI service that records every turn to SQLite.

```
┌──────────────┐   JSON/HTTP   ┌───────────────────────────────┐
│  index.html  │ ────────────► │  FastAPI  (uvicorn)           │
│  vanilla JS  │ ◄──────────── │   ├─ game engine (Bayesian)   │
└──────────────┘               │   ├─ classifier (PyTorch)     │
                               │   └─ SQLAlchemy → SQLite      │
                               └───────────────────────────────┘
                                        ▲
                        scikit-learn TF-IDF  +  torch.nn MLP
                        trained by  ml/prepare_data → ml/train
```

---

## Quick start

```bash
cd food-guess-game
./run.sh
```

`run.sh` installs dependencies if needed, validates the dish catalogue, trains the
model on first run (~2 minutes on a laptop CPU), then serves the game.

Open **http://127.0.0.1:8000** — API docs live at **/docs**.

Manual equivalent:

```bash
pip install -r requirements.txt
python3 -m ml.validate_catalog     # schema + clue-leak checks
python3 -m ml.prepare_data         # build the training corpus
python3 -m ml.train                # fit TF-IDF + train the network
uvicorn app.main:app --reload
```

---

## The three modes

**1 · The chef hides a dish.** Four hand-written clues per dish, ordered from
oblique to nearly-giving-it-away, then generated attribute hints once those run
out. Guesses are resolved by a four-stage fuzzy matcher, so `mapo dofu`,
`rosti` and `i think it's poutine` all land correctly. Points decay with every
clue, hint and wrong guess.

**2 · The chef reads your mind.** You think of a dish; the engine finds it by
asking yes/no questions. It does *not* keep a shrinking candidate list — it
keeps a full probability distribution over the catalogue and updates it by
Bayes' rule with an explicit answer-noise model. Each question is chosen to
maximise expected information gain:

```
IG(q) = H(belief) − [ P(yes)·H(belief | yes) + P(no)·H(belief | no) ]
```

Measured over the whole catalogue (`python3 -m tests.simulate`):

| player behaviour              | solved | mean questions |
|-------------------------------|--------|----------------|
| answers honestly              | 100 %  | 8.0            |
| 10 % of answers wrong         | 96.4 % | 10.9           |
| 20 % answered "not sure"      | 100 %  | 10.0           |
| 15 % wrong **and** 15 % unsure| 76.3 % | 14.0           |

Eight questions for 139 dishes is close to the information-theoretic floor of
log₂(139) ≈ 7.1. The soft update is what buys the robustness: a hard filter
would eliminate the true dish permanently on the player's first mistake.

**3 · Describe it to the model.** Free text straight into the classifier, with
the ranked top-5 and confidences. Wrong answers can be corrected in the UI, and
those corrections are stored as training signal.

---

## The model

`scikit-learn` handles feature extraction, `torch` does the learning.

- **Features** — a `FeatureUnion` of word 1–2 grams and character 3–5 grams
  (`char_wb`). The character view is not decoration: players type `biriyani`,
  `shakshouka`, `bolognaise`, and a word-only model treats every misspelling as
  an unseen token.
- **Network** — `Linear → LayerNorm → GELU → Dropout` ×2 into a softmax head.
  LayerNorm rather than BatchNorm because inference happens one sample at a time
  behind the API. ~4.7 M parameters at the default 12 000-dimensional input.
- **Training** — AdamW, cosine LR schedule, label smoothing 0.05, gradient
  clipping, per-sample loss weighting, early stopping on validation macro-F1.

Held-out results on the bootstrap corpus (139 classes):

| metric   | value |
|----------|-------|
| top-1    | 0.973 |
| top-3    | 0.989 |
| top-5    | 0.992 |
| macro-F1 | 0.973 |

**Read that honestly.** The held-out split comes from the same template bank
that generated the training rows, so it measures generalisation across the
template space, not across real human phrasing. The number to trust is what the
feedback loop reports once people have actually played
(`GET /api/analytics` → `classifier_feedback_accuracy`).

### Feeding it the Food.com dataset

The catalogue bootstrap exists so the project trains with zero downloads. Real
recipe text makes it materially better:

1. Download **Food.com Recipes and Reviews** from Kaggle.
2. Drop `recipes.csv` into `data/`.
3. `python3 -m ml.prepare_data && python3 -m ml.train`

`prepare_data` streams the CSV in chunks, matches recipe titles to catalogue
dishes through the alias index, parses the R-style `c("a","b")` list columns,
**strips the dish name out of the harvested text** so the model cannot cheat,
and weights those rows 3× against synthetic ones. It reports coverage and names
the dishes it found no match for.

### Learning from play

Every prediction and every correction is stored. Export them in corpus format:

```bash
curl -s localhost:8000/api/analytics/feedback/export > feedback.json
```

Append those rows to `data/artifacts/corpus.csv.gz`, retrain, then hot-swap the
new weights into the running server without a restart:

```bash
curl -X POST localhost:8000/api/predict/reload
```

---

## The catalogue

139 dishes across 18 regions in `data/dishes/*.json`, weighted towards
difficulty 3–5. Each entry carries structured attributes (course, protein, carb,
method, temperature, 29 boolean flags), 4–6 key ingredients, four progressive
clues and a post-reveal fact.

`ml/validate_catalog.py` enforces the schema and controlled vocabularies, checks
flags against the enum fields, and — the important one — verifies that **no clue
gives away its own answer**, including inside compound words. Clues 1–3 may not
contain the name or any alias; clue 4 is the deliberate near-giveaway and may use
a translated description but never the canonical name. Facts are exempt because
they are only ever shown after the reveal.

Adding a dish is just another object in any of those files; the question bank,
the corpus generator and the classifier all pick it up on the next run.

---

## Layout

```
app/
  main.py              FastAPI app, lifespan, static mount
  config.py            settings + game tuning constants
  database.py          engine, session factory, SQLite pragmas
  db_models.py         GameSession, Turn, Guess, Prediction, Feedback, DishStat
  schemas.py           Pydantic request/response contract
  routers/
    game.py            both play modes
    predict.py         classifier + feedback + hot reload
    analytics.py       aggregates and corpus export
  services/
    catalog.py         dish loading, filtering, the 82-question bank
    engine.py          Belief, MindReader, information gain, scoring
    inference.py       model loading, thread-safe predict, fallback
    matching.py        four-stage fuzzy guess resolution
ml/
  validate_catalog.py  schema + clue-leak checks
  prepare_data.py      corpus builder (bootstrap + Food.com)
  model.py             the nn.Module
  train.py             training loop, metrics, artifacts
data/dishes/*.json     the catalogue
frontend/index.html    the client, single file, no build step
tests/                 pytest suite + engine simulator
```

---

## API

| method | path | purpose |
|---|---|---|
| GET  | `/api/health` | status, catalogue size, model metrics |
| POST | `/api/game/clue/new` | start a clue round |
| POST | `/api/game/clue/next` | reveal the next clue or hint |
| POST | `/api/game/clue/guess` | submit a guess |
| POST | `/api/game/clue/giveup` | reveal the answer |
| POST | `/api/game/mind/new` | start a mind-reading round |
| POST | `/api/game/mind/answer` | answer yes / no / unsure |
| POST | `/api/game/mind/verdict` | confirm or reject the engine's guess |
| GET  | `/api/game/regions` | filter options |
| POST | `/api/predict` | classify a free-text description |
| POST | `/api/predict/feedback` | record the true answer |
| POST | `/api/predict/reload` | hot-swap retrained weights |
| GET  | `/api/analytics` | win rates, hardest dishes, model accuracy |
| GET  | `/api/analytics/feedback/export` | corrections in corpus format |

---

## Tests

```bash
python3 -m pytest tests -q          # 25 tests
python3 -m tests.simulate --noise 0.1   # engine against every dish
```

The suite covers catalogue integrity (including a check that no two dishes share
an identical answer signature across all 82 questions — if they did, the mind
reader could never separate them), fuzzy matching, Bayesian convergence,
robustness to a lying player, and every endpoint.

---

## Configuration

Set via environment variables:

| variable | default | meaning |
|---|---|---|
| `FGG_DATABASE_URL` | `sqlite:///data/game.db` | any SQLAlchemy URL; PostgreSQL works unchanged |
| `PORT` / `HOST` | `8000` / `127.0.0.1` | used by `run.sh` |

Game balance (question limits, guess threshold, answer-noise probabilities,
scoring) lives in `app/config.py`.
