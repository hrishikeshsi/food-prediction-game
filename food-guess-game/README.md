# Chef Mystico — a bidirectional food prediction engine

A full-stack guessing game over a curated catalogue of 350 dishes. One table of
yes/no questions about those dishes drives two engines that run in opposite
directions:

- **Forward — the chef reads your mind.** You think of a dish; the engine asks
  questions and names it. Its first guess is right at least 95% of the time for
  a player who gets up to one answer in ten wrong.
- **Reverse — you guess the chef's dish.** The engine hides a dish and reveals
  hints, each ruling out about half of what is left, until you name it.

```
┌──────────────────────┐  JSON/HTTP  ┌─────────────────────────────────────────┐
│ index.html (the app) │ ──────────► │ FastAPI (uvicorn)                       │
│ tree.html            │ ◄────────── │  ├─ forward: Bayesian decision engine   │
│                      │             │  ├─ reverse: hint ladder on a fitted    │
│ vanilla JS, no build │             │  │           scikit-learn decision tree │
└──────────────────────┘             │  ├─ free-text classifier (PyTorch)      │
                                     │  └─ SQLAlchemy → SQLite                 │
                                     └─────────────────────────────────────────┘
                 one question table: 183 questions × 350 dishes
```

---

## Quick start

```bash
cd food-guess-game
./run.sh
```

`run.sh` creates a virtualenv in `.venv` if there is none, installs the
dependencies, validates the dish catalogue, trains the free-text classifier on
first run (about two minutes on a laptop CPU), then serves the game.

Open **http://127.0.0.1:8000** and pick a mode. The decision tree is the
**Culinary Tree** tab on the main page (or **/tree** on its own), and the API
docs are at **/docs**.

The front end also works opened straight from `frontend/index.html`, or from an
editor preview server such as VS Code's Live Server: it looks for the game
server at `127.0.0.1:8000`, and waits for it if it is not running yet.

The two game modes do not need the classifier. To skip the PyTorch download and
training and go straight to playing:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scikit-learn joblib fastapi "uvicorn[standard]" pydantic sqlalchemy
.venv/bin/python -m uvicorn app.main:app
```

---

## Forward mode: the chef reads your mind

The engine is a decision tree grown one node at a time, with soft splits.

- **Belief, not a shortlist.** It keeps a probability for every dish. Each answer
  multiplies a dish's probability by how likely that answer is for that dish,
  then renormalises (Bayes' rule). A dish that disagrees with an answer is
  demoted, never deleted, so one mistaken answer cannot lose it for good.
- **Question choice.** At every step it asks the question with the highest
  expected information gain over the current belief, the same criterion a
  decision tree uses to choose a split:

  ```
  gain(q) = h( Σ_d belief_d · P(yes | q, d) )  −  Σ_d belief_d · h( P(yes | q, d) )
  ```

  with `h` the binary entropy. The first term rewards an even split; the second
  subtracts the part of the answer that is only noise.
- **Stopping rule.** It names a dish once that dish holds 95% of the
  probability, or when nothing left is worth asking, or at 40 questions. A guess
  made at 95% confidence is right 95% of the time as long as the answer model
  fits the player. That is what the accuracy target rests on.
- **Answer model.** A player's answer is assumed to match the catalogue 90% of
  the time. Ingredient and regional-cuisine questions are trusted a little less.
  "Not sure" teaches the engine nothing and costs one question.

Measured by playing every dish 20 times against scripted players who answer
from the catalogue, with a set share of answers flipped or replaced by "not
sure" (7,000 rounds per row; `notebooks/benchmark.ipynb`):

| player | first guess right | questions (mean) | hard tree | previous engine |
|---|---|---|---|---|
| answers honestly | 100.0% | 11.4 | 100.0% | 96.4% |
| 5% of answers wrong | 99.2% | 13.9 | 65.2% | 77.1% |
| **10% of answers wrong** | **95.9%** (95% interval 95.4–96.3) | 17.4 | 42.2% | 56.7% |
| *stress:* 20% "not sure" | 99.8% | 14.6 | 40.9% | 89.7% |
| *stress:* 10% wrong + 10% "not sure" | about 95% | 19.4 | 28.6% | 55.0% |
| *stress:* 15% wrong | 86.5% | 22.0 | 25.6% | 38.0% |

The first three rows are the target; the engine is within a question and a half
of the information-theoretic floor on the third (15.9 questions). The combined
stress row moves between 94.5% and 95.4% from one run to the next, so read it
as at the line, not above it. *Hard tree*
is the scikit-learn tree below, walked once with the same answers. *Previous
engine* is this project before the redesign: 82 questions, a guess at 55%
confidence, a cap of 20.

**What the table does not show.** The scripted player's mistakes are
independent coin flips. A real player's come from one wrong belief about the
dish and repeat across related questions, so expect live accuracy to be lower
until the answer model is fitted to real play. The server logs every answer for
that purpose: `GET /api/analytics/answer-noise` reports how often players'
answers actually matched the catalogue, by kind of question, next to the values
assumed in `app/config.py`.

When the engine loses, it asks what the dish was and shows which of your
answers disagreed with its catalogue.

---

## Reverse mode: you guess the chef's dish

The hints for a hidden dish are that dish's root-to-leaf path through a fitted
decision tree, read out as statements. Forward mode asks those questions; here
the chef answers them unprompted, so every answer is true.

Each node of the tree is the most even split of the dishes still in play, so
each hint removes about half of them: 350 → 180 → 94 → 45 → 23 → 12 → 6 → 3 → 2
→ 1 for the median round. Broad traits come first and the distinguishing detail
last. Over all 350 dishes and all six trees, no dish is pinned down before the
sixth generated hint, and no hint contains part of the dish's own name.

The four hand-written clues each dish carries are woven around that ladder:

```
clue 1           an oblique opener
traits …         the ladder, until four or fewer dishes still fit
clue 2           the handover
traits …         the rest of the ladder, down to one dish
clue 3, clue 4   the signature details; clue 4 all but names it
```

A wrong guess changes what comes next. If the guessed dish contradicts a hint
already given, the chef points back at that hint. Otherwise the next hint is
the split where the guess and the hidden dish part ways in the tree: the one
fact that rules the guess out (“Unlike Bigos: It includes a starch or grain.”).

A trait costs fewer points than a written clue, because it says less.

---

## The decision tree

`app/services/tree.py` fits scikit-learn's `DecisionTreeClassifier` with the
entropy criterion on the question table: one row per dish, one boolean feature
per question, the dish itself as the class.

| | |
|---|---|
| leaves | 350, one per dish |
| questions to reach a dish | 8.56 on average, 10 at most |
| theoretical minimum | 8.45 (log₂ 350) |
| dishes reached by honest answers | 100% |

It is used in three places: reverse mode walks it for hints, the notebook uses
it as the hard-split baseline, and it is there to look at.

```bash
python3 -m ml.decision_tree      # writes data/artifacts/tree/
```

writes the fitted classifier (`.joblib`), a readable outline (`.txt`), Graphviz
source (`.dot`) and the nested JSON the **/tree** page draws. That page draws
the tree in 3D, with every dish as a leaf at a branch tip: drag to turn it, and
hover or click a leaf to see the questions that lead to that dish.

Tree 0 is the canonical tree. Reverse mode also fits five randomised variants
(each split considers a random 40% of the questions) so that successive rounds
do not open with the same hint.

---

## The question bank

183 yes/no questions in `app/services/catalog.py`, all derived from fields the
catalogue already had:

| kind | count | source |
|---|---|---|
| attribute | 65 | flags, course, protein, carb, method, temperature |
| origin | 54 | region group, region, country |
| cuisine | 16 | regional cuisines with at least three dishes |
| ingredient | 48 | curated ingredient concepts matched against `key_ingredients` |

Three filters run when the bank is built: a question must split the catalogue;
a generated question must be true of at least two or three dishes; and a
question whose answers duplicate an earlier one is dropped, because the engine
would count the re-ask as fresh evidence.

Every question carries two declarative sentences as well as its interrogative,
which is what lets reverse mode state it as a hint.

The earlier 82-question bank left 20 dishes in 8 groups impossible to tell apart
(Naan / Garlic Naan / Tandoori Roti / Kulcha / Bhakri, for one). The current bank
has no such groups, and the typical dish is separated from its nearest
neighbour by 8 answers instead of 4.

---

## The free-text classifier

A third interface, `POST /api/predict`: describe a dish in words and get a
ranked top five. It can also seed forward mode with a prior (`hint` on
`/api/game/mind/new`).

`scikit-learn` handles feature extraction, `torch` does the learning.

- **Features** — a `FeatureUnion` of word 1–2 grams and character 3–5 grams
  (`char_wb`), so that `biriyani`, `shakshouka` and `bolognaise` are not unseen
  tokens.
- **Network** — `Linear → LayerNorm → GELU → Dropout` ×2 into a softmax head.
- **Training** — AdamW, cosine LR schedule, label smoothing 0.05, gradient
  clipping, per-sample loss weighting, early stopping on validation macro-F1.

Its held-out metrics are written to `data/artifacts/metrics.json` by
`python3 -m ml.train` and served by `GET /api/health`. The held-out split comes
from the same template bank that generated the training rows, so it measures
generalisation across templates, not across real phrasing; the number to trust
is `classifier_feedback_accuracy` from `GET /api/analytics` once people have
played. Without trained artifacts the endpoint falls back to keyword overlap.

To train on real recipe text, download **Food.com Recipes and Reviews** from
Kaggle, put `recipes.csv` in `data/`, and run
`python3 -m ml.prepare_data && python3 -m ml.train`. Corrections made in play
are exported by `GET /api/analytics/feedback/export`, and
`POST /api/predict/reload` hot-swaps retrained weights.

---

## The catalogue

350 dishes across 18 regions in `data/dishes/*.json`. Each entry carries
structured attributes (course, protein, carb, method, temperature, 29 boolean
flags), three to six key ingredients, four progressive clues and a post-reveal
fact.

`ml/validate_catalog.py` enforces the schema and controlled vocabularies, checks
flags against the enum fields, and verifies that no clue gives away its own
answer, including inside compound words. The same leak check is applied to
every generated hint.

Adding a dish is another object in any of those files. The question bank, both
engines, the decision trees and the classifier corpus pick it up on the next
start.

---

## Layout

```
app/
  main.py              FastAPI app, lifespan, page routes
  config.py            settings: thresholds, answer model, scoring
  database.py          engine, session factory, SQLite pragmas
  db_models.py         GameSession, Turn, Guess, Prediction, Feedback, DishStat
  schemas.py           Pydantic request/response contract
  routers/
    game.py            both game modes
    tree.py            the decision tree and engine settings, for inspection
    predict.py         classifier + feedback + hot reload
    analytics.py       aggregates, observed answer noise, corpus export
  services/
    catalog.py         dish loading, filtering, the question bank
    engine.py          forward mode: belief, information gain, stopping rule
    tree.py            the scikit-learn decision trees
    hints.py           reverse mode: the hint ladder
    inference.py       classifier loading, thread-safe predict, fallback
    matching.py        four-stage fuzzy guess resolution
ml/
  benchmark.py         every benchmark figure, shared by the CLI and the notebook
  decision_tree.py     fit and export the tree
  validate_catalog.py  schema + clue-leak checks
  prepare_data.py      classifier corpus builder
  model.py, train.py   the classifier
notebooks/
  benchmark.ipynb      the evaluation, with outputs
data/dishes/*.json     the catalogue
frontend/
  index.html           the app: welcome, both game modes, Culinary Tree · About · Help · Future scope
  styles.css, app.js   its design system and logic; api.js finds the server
  tree.html            the Culinary Tree, the decision tree in 3D (three.js); framed by index.html
  assets/hero.mp4      the gameplay video (see "The front end")
../data/               chef-guess.html, you-guess.html — old addresses, now redirects
tests/                 pytest suite + benchmark CLI
```

---

## API

| method | path | purpose |
|---|---|---|
| GET  | `/api/health` | status, catalogue size, classifier metrics |
| POST | `/api/game/mind/new` | forward: start a round |
| POST | `/api/game/mind/answer` | forward: answer yes / no / unsure |
| POST | `/api/game/mind/verdict` | forward: confirm or reject the engine's guess |
| POST | `/api/game/mind/reveal` | forward: give up and say what the dish was |
| POST | `/api/game/clue/new` | reverse: start a round |
| POST | `/api/game/clue/next` | reverse: reveal the next hint |
| POST | `/api/game/clue/guess` | reverse: submit a guess |
| POST | `/api/game/clue/giveup` | reverse: reveal the answer |
| GET  | `/api/game/regions` | filter options |
| GET  | `/api/tree` | the canonical decision tree as JSON, with summary figures |
| GET  | `/api/tree/text` | the same tree as an indented outline |
| GET  | `/api/tree/engine` | thresholds, answer model, question counts |
| POST | `/api/predict` | classify a free-text description |
| POST | `/api/predict/feedback` | record the true answer |
| POST | `/api/predict/reload` | hot-swap retrained weights |
| GET  | `/api/analytics` | win rates, hardest dishes, classifier accuracy |
| GET  | `/api/analytics/answer-noise` | how players' answers compare with the catalogue |
| GET  | `/api/analytics/feedback/export` | corrections in corpus format |

---

## The front end

One page, three screens, no build step:

- **Welcome** — pure black, the line-art Craving Jadugar emblem (it draws
  itself, follows the cursor, and casts a spell every few seconds), and the two
  ways to play.
- **The arena** — both game modes over a looping video, behind a dark scrim and
  frosted-glass panels. Every round opens by checking the player is ready ("Do
  you have a dish in your mind?" / "I have a dish in my mind. Are you ready to
  guess?"). The chef from the welcome page hosts: his lines type into a speech
  bubble while his mustache moves, and he nods at answers, shakes his head at
  wrong guesses, tilts it while the server thinks, and casts a spell as each
  dish is unveiled from under a lifting cloche.
- **About · Help · Future scope** — linked from the welcome page.
- **Sound** — synthesised in the browser by `frontend/sound.js`, no audio files: a
  heavy, muffled thud on every click or tap (hovering is silent), the cloche
  landing and lifting with a deep boom, a shimmer for the chef's spells, a
  dull double thud for a wrong guess, and on the welcome page a tiny glassy
  spark for each speck of stardust behind the cursor. Nothing plays before the visitor's first
  click; the speaker button in the bottom-left corner turns it off.

Palette: black `#000000` / `#09090B`, charcoal `#121215`, white `#FAFAFA`, and
champagne gold `#C8A96E` kept to hairlines and badges. Type: Instrument Serif,
Plus Jakarta Sans, JetBrains Mono. Everything respects the system's reduced-motion
setting.

**To use your own gameplay video**, put the file in `frontend/assets/` and change
the `<source src="assets/gameplay.mp4">` line in `frontend/index.html` (it is
marked `GAMEPLAY VIDEO`). Any `https://` address works too, and a WebM line above
it is used by browsers that support WebM. Short, compressed clips load fastest:
`gameplay.mp4` is the 16-second 4K `hero.mp4` re-encoded at 1080p, 9 MB instead
of 87, for example with
`ffmpeg -i hero.mp4 -an -vf scale=1920:-2 -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p -movflags +faststart gameplay.mp4`.

Each mode has its own address: `/#forward`, `/#reverse`, `/#about`, `/#help`,
`/#future`.

---

## Live deployment

Two free pieces:

- **The page, on GitHub Pages** —
  [hrishikeshsi.github.io/food-prediction-game](https://hrishikeshsi.github.io/food-prediction-game/).
  `.github/workflows/pages.yml` publishes `frontend/` whenever a push to `main`
  changes it. One-time setup: repository **Settings → Pages → Source: GitHub
  Actions**.
- **The game server, on Render** — described by `render.yaml` at the repository
  root: on render.com, **New → Blueprint →** this repository **→ Apply**. It
  installs `requirements-server.txt` (no PyTorch) and serves the API, and the
  whole game as well, at `https://craving-jadugar.onrender.com`.

The page finds its server by itself: the same address when the server serves it,
`127.0.0.1:8000` when it is opened on your own machine, and the Render server
everywhere else. If Render gives the service a different address, change
`LIVE_API` at the top of `frontend/api.js`.

The free Render plan sleeps after 15 idle minutes, and the first visit after
that takes up to a minute — the page says it is waking the kitchen meanwhile.
Game history lives in SQLite on the server's own disk, so it starts afresh
whenever the service is redeployed.

---

## Tests and benchmarks

```bash
python3 -m pytest tests -q            # 43 tests
python3 -m tests.simulate             # forward mode against every scripted player
python3 -m tests.simulate --legacy    # the previous engine, for comparison
python3 -m tests.simulate --tree      # the hard decision tree
python3 -m tests.simulate --ladder    # reverse mode: how the hints narrow the field
```

The suite covers catalogue integrity, the question bank, fuzzy matching, the
information-gain arithmetic, the accuracy target itself, tree purity, hint
plans (true, leak-free, never an early giveaway), every endpoint, and the page
routes.

`notebooks/benchmark.ipynb` has the full evaluation with tables and charts. To
re-run it, `pip install matplotlib nbconvert ipykernel` and open it in Jupyter.

---

## Configuration

Set via environment variables:

| variable | default | meaning |
|---|---|---|
| `FGG_DATABASE_URL` | `sqlite:///data/game.db` | any SQLAlchemy URL; PostgreSQL works unchanged |
| `PORT` / `HOST` | `8000` / `127.0.0.1` | used by `run.sh` |
| `PYTHON` | `.venv/bin/python` | interpreter `run.sh` should use |

Engine behaviour lives in `app/config.py`: the guess threshold, the question
cap, the answer model for each kind of question, the number of hint trees, and
scoring.
