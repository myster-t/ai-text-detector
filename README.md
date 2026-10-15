# ai-text-detector

Is this text machine-generated? Run several independent, fully local checks and get a graphical, printable report per file
that shows where the text sits against known-human and known-machine control texts on every measure.

1. **Binoculars** (Hans et al., ICML 2024): the best-regarded zero-shot detector, using the paper's exact Falcon-7B model pair
   and thresholds.
2. **Fast-DetectGPT** (Bao et al., ICLR 2024), analytic form, computed from the same two forward passes.
3. **Token statistics** from a single small model: GLTR rank buckets (Gehrmann et al., 2019), the log-likelihood/log-rank
   ratio (DetectLLM, Su et al., 2023), mean log-rank and next-token entropy, plus a GLTR-style token heat map of the text.
4. **Perplexity and burstiness** (the GPTZero family) under two language models, sentence by sentence.
5. **Stylometry** without any model: lexical diversity (MATTR, MTLD, hapax ratio), sentence- and paragraph-length variation,
   punctuation and construction rates (em dashes, "not X, but Y", rule-of-three lists, contractions, first person).
6. **Vocabulary and construction tells, each labelled with the era it was measured in**: the 2024 PubMed excess-vocabulary
   list (Kobak et al., MIT), the 21 "delve" words (Juzek & Ward, 2025), Wikipedia's "Signs of AI writing" core list
   (2025-26, CC BY-SA), the slop-forensics essay lists and antislop phrase lists (Paech, 2025, MIT / Apache 2.0), and the
   construction frames that still survive in 2026 models (corrective "not just X but Y", "rather than merely",
   "this matters", "adds another layer"), plus punctuation diversity. Rates per thousand words, every match marked in the
   text.
7. **A supervised vote**: a 2026 ModernBERT classifier (MIT) trained on 11 million RAID-era texts, used as one capped vote.
8. **Writing-pattern and hidden-character checks**: AI-isms such as "delve", "in today's fast-paced world" and
   "in conclusion" (rules from the avoid-ai-writing project), plus zero-width characters, bidi controls and lookalike letters
   from other alphabets, each marked inline in the text.

All of it feeds one **combined score and concern level** (low / moderate / high), computed as the evidence-weighted average of
every measure that actually separates the bundled controls, validated leave-one-out on those controls.

No API keys, no uploads, nothing leaves your machine. The installer assumes nothing: no Homebrew, no Python, no Node.

## Quick start

```sh
git clone https://github.com/myster-t/ai-text-detector.git
cd ai-text-detector
./install.sh                    # installs a Python runtime, the packages and ~36 GB of current models; ends with a self-test
./detect essay.txt              # writes ~/Downloads/261015-ai-report-1201-essay.html (date, then time)
./install.sh --older-models     # optional: the 2023 Falcon-7B / Qwen2.5 set with the Binoculars paper's thresholds (~33 GB)
./detect essay.txt --older-models
```

The default model set is chosen by a bake-off on the 67 bundled controls among current, fully open, non-gated models: the
OLMo-3 7B base/instruct pair (Ai2, 2025-26) for the pair-based scores, the Qwen3.5 small base models for the token
statistics, and a 2026 ModernBERT classifier as one vote. It has no published Binoculars thresholds, so the report never
says "AI" or "human" outright; it gives a concern level from the measures that separate the controls and shows where the
text falls on every measure. `--profile smollm3` is a light set (about 13 GB) that scores almost as well; `--older-models`
is the original 2023 Falcon / Qwen2.5 set with the paper's thresholds; `--profile qwen35` is the all-Qwen3.5 set. Any
combination can be installed side by side.

`./detect` takes any number of text files or Google Docs links and writes **one self-contained HTML file per run** into
`~/Downloads` (or `./reports` if there is no Downloads folder), named `<yymmdd>-ai-report-<hhmm>-<first text>.html` (for example `261015-ai-report-1201-Gary.html`), so
no run ever overwrites another and files sort by date. The file opens with an overview of every text and its concern
level, a sticky navigation bar, and then one collapsible section per text (or per Google Docs tab) with the full report.
`--out DIR` changes the folder; `--out path.html` sets the exact file. Also `--fast`, `--low-memory` and `--profile NAME`
(see Memory).

**Google Docs.** Pass a document link instead of a file and every tab becomes its own report:

```sh
./detect "https://docs.google.com/document/d/<id>/edit?tab=t.0"          # all tabs it can find; this tab if only one
./detect "https://docs.google.com/document/d/<id>/edit" --tabs t.0,t.k8x2   # name the tabs (the tab= value in the browser URL)
```

This uses Google's plain-text export, which needs no account or API key but requires the document to be shared as
"Anyone with the link" (Share > General access); a private document yields a clear message saying so. The tool reads the
tab list from the document's preview page (served without sign-in for link-shared documents), exports each tab on its own,
and names each report by the document title, the tab number and the tab's opening words. The fetched text of each tab is
saved next to its report, so what was scored is on record. `--tabs` overrides discovery with explicit ids; if neither
yields anything, the whole document is scored as one text. Quote the link in the shell: `?` and `&` are shell characters.

To update later, run `./update.sh` inside the folder: it downloads the latest files from GitHub and copies them over this
folder (no git needed; a git clone is pulled instead), keeps your models, Python environment, settings, reports and any
control texts you added, then re-runs the installer, which only fetches what is newly required. `./update.sh --no-selftest`
skips the final self-test. The models load for the run and are released when it
exits; nothing stays resident. The control texts are scored once per model set and cached in `controls/scores/<profile>.json`
(committed to the repo, keyed by each control's text hash and the profile's model names), so a normal run scores only your
files; a control is re-scored only when its text or the model set changes, or when you pass `--rescore-controls`. Everything lives inside the checkout (`.venv/`, `models/`), so deleting the folder removes it all.

Requirements: macOS or Linux (Windows via WSL2) and a machine with 32 GB or more of memory for the 7B to 9B model pairs
(Apple Silicon or an NVIDIA GPU; CPU works but is slow). A run on an Apple M5 Max takes two to three minutes, most of it
loading weights.

### Memory

The default profile loads a 7B observer and a 7B performer together (about 29 GB of weights in bf16), so it wants a machine
with 32 GB or more. Smaller machines have three options, and the tools pick sensible ones on their own:

| Machine | What happens | Memory | What you lose |
| --- | --- | --- | --- |
| 32 GB or more | full default profile | ~30 GB peak | nothing |
| under 24 GB, any profile | `./detect` switches itself to `--low-memory`: the observer runs over your files first, its probabilities are kept in fp16, then the performer runs; peak is one model at a time | ~15 GB for the 7B pair, ~7 GB for `smollm3` | nothing; same numbers, a little slower |
| 16 GB | `./install.sh` installs the light profile `smollm3` instead (13 GB of models, scored almost as well in the bake-off) and `./detect` uses it by default | ~7 GB peak | Binoculars 0.83 → 0.81, Fast-DetectGPT 0.76 → 0.79 on the controls |
| anything, in a hurry | `./detect --fast file.txt` skips the pair entirely: small base models, classifier and deterministic checks only | ~5 GB, about 20 seconds | Binoculars, Fast-DetectGPT and alignment imprint; the strongest measures (token statistics, classifier, shuffle shift) remain |

Control scores for every profile ship in `controls/scores/`, so a small machine never has to score the 67 controls; it only
scores your files.

### Disk space

The models are the cost. They are downloaded once into `./models` (or wherever `--models-dir` points) and never duplicated.

| What | Size on disk |
| --- | ---: |
| Python environment (`.venv`, PyTorch and Transformers) | about 3 GB |
| default set (`olmo3`): Olmo-3-1025-7B, Olmo-3-7B-Instruct, Qwen3.5-2B-Base, Qwen3.5-0.8B-Base, ModernBERT classifier | 36 GB |
| `--profile smollm3` (light): SmolLM3-3B-Base, SmolLM3-3B, the same small models and classifier | 13 GB |
| `--profile qwen35`: Qwen3.5-9B-Base, Qwen3.5-9B, the same small models and classifier | 45 GB |
| `--older-models` (`original`): Falcon-7B, Falcon-7B-Instruct, Qwen2.5-1.5B, GPT-2 | 33 GB |
| Everything | about 120 GB |

The installer checks free space at the models location before downloading. To put the models on another disk:

```sh
./install.sh --models-dir /Volumes/External/aitd-models      # remembered in .models-dir for every later ./detect
```

`AITD_MODELS_DIR` in the environment overrides that file. To remove what was installed:

```sh
./uninstall.sh --profile hybrid     # one profile's models (models shared with other profiles are kept)
./uninstall.sh --all-models         # every model
./uninstall.sh --all                # models and the Python environment
./uninstall.sh --profile original --dry-run   # preview what a removal would free
```

Nothing is installed outside the checkout except uv itself (in `~/.local/bin`), which other tools may use; remove it with
`rm ~/.local/bin/uv ~/.local/bin/uvx` if you want it gone too.

## Expected results

Every run re-scores 67 bundled controls (35 human, 32 machine; see [`controls/README.md`](controls/README.md)) with the same
models as your file, and every report draws your file among them on every measure. The two anchor controls score like this
under the `--older-models` set, the only one with published thresholds:

| Text | Binoculars | Verdict | Qwen2.5-1.5B perplexity | Burstiness | GPT-2 perplexity | AI-ism score |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `controls/human/stewart-letter-1913.txt` (1913 letter, public domain) | 0.972 | human | 24.8 | 43.8 | 47.5 | 1 / 100 |
| `controls/ai/claude-marketing-paragraph.txt` (unedited LLM output) | 0.800 | AI | 4.9 | 10.0 | 15.5 | 12 / 100 |

Under the default hybrid set the pair-based scores are the same as in that table (same Falcon pair) but no verdict line is
drawn; the self-test checks that the human anchor scores above the machine anchor and that the group means separate. The index page of
every run lists all measures ranked by how well they separate the controls.

How to read them:

- **Binoculars**: below **0.9015** reads as machine-generated at the paper's accuracy-optimal setting, below **0.8536** at its
  low-false-positive setting. Higher is more human-like.
- **Perplexity**: machine text in the controls sits near 5 with burstiness near 10; human prose runs far higher on both.
- **AI-ism score**: 0 to 100 from the pattern rules. Note it reads the LLM control as "minimal AI signals" (12): generic LLM
  prose can be free of the obvious tells, which is why this check is editorial evidence, not a detector.

In testing on nine polished personal essays written by a student, Binoculars scored 1.04 to 1.28 and perplexity 24 to 107,
well on the human side, while a commercial-style trained classifier had flagged all nine at the edge of its threshold. That
disagreement is the reason this repo runs several methods and shows controls beside every result.

A rendered report for both anchor controls is [`examples/report.html`](examples/report.html).

## Model profiles

`profiles.json` defines the model sets. `./detect`, `./install.sh` and `./uninstall.sh` take `--profile NAME`;
`--older-models` is shorthand for `--profile original`, and no flag means `olmo3`.

| Profile | Binoculars pair | Perplexity models | Thresholds |
| --- | --- | --- | --- |
| `olmo3` (default) | allenai/Olmo-3-1025-7B, Olmo-3-7B-Instruct | Qwen3.5-2B-Base, Qwen3.5-0.8B-Base | none; best current pair measured |
| `smollm3` (light) | HuggingFaceTB/SmolLM3-3B-Base, SmolLM3-3B | same | none; 13 GB in total |
| `qwen35` | Qwen/Qwen3.5-9B-Base, Qwen3.5-9B | same | none; all-Qwen3.5 option |
| `hybrid` | tiiuae/falcon-7b, falcon-7b-instruct | same | none; 2023 pair with current small models |
| `original` (`--older-models`) | tiiuae/falcon-7b, falcon-7b-instruct | Qwen2.5-1.5B, GPT-2 | the paper's; reports state a verdict |
| `falcon3` | tiiuae/Falcon3-7B-Base, Falcon3-7B-Instruct | same | none; bake-off candidate, measured worse (see below) |
| `dev` | Qwen3.5-2B-Base, Qwen3.5-0.8B-Base | GPT-2 | none; small models for working on the report itself, not a detector |

Binoculars' published thresholds were derived for the Falcon pair and mean nothing for any other, so the default profile
deliberately draws no lines and makes no call. Its reports show where your file falls on each measure relative to the 21
controls, as dots on a number line, and say which side of the gap between the human and machine groups it lands on.

```sh
./install.sh --older-models                   # downloads the 2023 set (33 GB); the default set stays if present
./detect --older-models essay.txt
```

Profiles are plain entries in `profiles.json` (observer, performer, perplexity models, optional thresholds). Add one to try
any other pair; the installer, runner and uninstaller all read the same file, so a new profile needs no code changes.

## What a report contains

- An **overview** at the top: every text in the run with its concern level and combined score, links to each section, the
  control number lines, the full comparison table, and (collapsed) the ranking of measures by how well they separate the
  controls.
- Per text, an **overall assessment**: a concern level (low, low to moderate, moderate, high) and a combined score with its position
  among the human and machine controls; under `--older-models` also the Binoculars verdict against the published
  thresholds; plus a count of how many directional measures put the file on the human or machine side of the control
  cut-point, overall and among the strong measures.
- **Which tests say what**: a table of every test with the year it was designed, its source, what this file scored, which
  side of the control cut-point that lands on, and the test's AUC on the controls; weak or undirected tests are collapsed
  below it. A violin of the combined score against the controls' leave-one-out scores sits under the table.
- Summary tiles (Binoculars, Fast-DetectGPT, perplexity, GLTR top-10 share, AI-isms, sentence-length variation, hidden
  characters), each with a bar showing the human and machine control ranges and the file's position.
- **The shape of the text**: violin panels, split into **main factors** (the measures with AUC at least 0.80 on the controls
  under this model set, at least nine) and a collapsed **secondary** set, human controls in green and machine controls in
  red with the individual control points and medians, the cut-point that best separates the two groups as a dotted line,
  and the file as a line across both. The axis always runs with the human direction up; the file line and the panel border
  are green or red for the side of the cut the file lands on; each panel states the AUC with which that measure alone
  separates the controls, and panels under 0.7 are drawn grey, dashed and desaturated as weak evidence, so the red panels
  are the ones to read.
- Number-line strips for Binoculars and Fast-DetectGPT with threshold lines drawn only when they apply.
- A **token heat map**: the text with every token coloured by where the real next token sat in the model's ranking
  (top-1, top-10, top-100, top-1000, beyond); hover for rank, log-probability and entropy.
- **Provenance signals**: no-break spaces and whether they sit directly after sentence ends (a pattern typical of text
  copied from a chat window or word processor), narrow no-break spaces (the character ChatGPT emitted in spring 2025),
  zero-width characters, curly versus straight quotes, em dashes versus typed double hyphens, the ellipsis character, line
  separators and soft hyphens, each with what it usually indicates. These say where the text was composed or copied from,
  not who wrote it.
- The **annotated text**: every AI-ism match marked inline by severity, slop-list matches dotted, sentence spans the pattern detector flags, each
  hidden or non-standard character shown as a red code-point badge in place, and mixed-script words outlined. The pattern
  and hidden-character tables below it give positions and context.
- A sentence-by-sentence perplexity table with bars, showing where a text is unusually predictable.
- Every measurement for the file and every control in a collapsed table (click to open), and a methods and limits section.
- The index page lists all files, the control table, and a ranking of all measures by how well they separate the controls.

The report is a single self-contained HTML file (no scripts, fonts or external assets), prints to A4, and renders in light
and dark mode; a run over ten texts is a few megabytes.

## Layout

```
detect, install.sh, update.sh, uninstall.sh   the four commands
profiles.json, requirements.txt               model sets and Python packages
aitd/                                         the code (report.py is the entry point; each check is also a standalone script)
controls/                                     the 67 control texts, their manifest, and the cached scores per profile
third_party/                                  vendored rule and vocabulary lists with their licences
examples/                                     a rendered report (both anchor controls)
scripts/                                      maintenance scripts (refresh the vendored style rules)
models/, .venv/, reports/                     created by the installer and by runs; not part of the repo
```

## Building blocks

Each check can run on its own and prints JSON:

```sh
.venv/bin/python aitd/binoculars.py draft.txt
.venv/bin/python aitd/perplexity.py Qwen/Qwen2.5-1.5B draft.txt
.venv/bin/python aitd/lmfeat.py Qwen/Qwen2.5-1.5B draft.txt     # GLTR buckets, log-rank, LRR, entropy
.venv/bin/python aitd/features.py draft.txt                      # stylometry, no model
.venv/bin/python aitd/style.py draft.txt
.venv/bin/python aitd/unicode_check.py draft.txt
```

```sh
.venv/bin/python aitd/slop.py draft.txt                          # slop vocabulary rates and matches
.venv/bin/python aitd/classifier.py draft.txt                    # supervised classifier P(AI)
```

`aitd/charts.py` holds the SVG builders (violins, strips, position bars, token heat map) and has no dependencies.

The scripts find the models in `./models` and run offline by default; no environment variables are needed.

## How it works

**Binoculars** scores a text as the ratio of its perplexity under a "performer" model to the cross-perplexity between an
"observer" model and the performer. The cross-perplexity measures how surprising the text is to one language model reading
another's predictions, which normalises away prompts and topics that make text predictable for reasons other than being
machine-written. Texts that a model finds much more predictable than this baseline score low. This repo uses the paper's
reference pairing, `tiiuae/falcon-7b` as observer and `tiiuae/falcon-7b-instruct` as performer, and its published thresholds.
`aitd/binoculars.py` is about 40 lines and follows Section 3 of the paper.

**Fast-DetectGPT** asks whether the text sits at a local maximum of the model's probability: machine text, being sampled
from a model, is more probable than the alternatives the model would itself have produced at each position; human text is
not. The analytic form (Bao et al., 2024, Section 3.2) needs no perturbation: with a sampling distribution from the observer
and log-probabilities from the performer, the score is the actual tokens' total log-probability minus its expectation under
sampling, divided by the standard deviation of that expectation. Higher is more machine-like. It reuses the two forward
passes Binoculars already makes, so it costs nothing extra.

**Token statistics** come from one forward pass of the first perplexity model. For each token the report keeps the rank of
the real token in the model's prediction, its log-probability and the entropy of the prediction. GLTR (Gehrmann et al., 2019)
summarises the ranks as the share of tokens in the top-10, top-100 and top-1000, which the report also paints over the text.
DetectLLM's log-likelihood/log-rank ratio (Su et al., 2023) divides mean negative log-probability by mean log-rank; machine
text scores higher. Mean entropy says how open the model's prediction was at each step; generators mostly follow the model in
low-entropy contexts, so human text averages higher.

**Stylometry** (`aitd/features.py`) needs no model: moving-average type-token ratio (Covington and McFall, 2010), MTLD (McCarthy
and Jarvis, 2010), hapax ratio, sentence- and paragraph-length coefficient of variation, word length, and per-thousand-word
rates of em dashes, semicolons, colons, questions, exclamations, contractions, first-person pronouns, "not X, but Y"
constructions and three-item lists. These describe register and rhythm; they shift with genre and separate the controls
only partly, which the report shows honestly through each panel's AUC.

**Slop vocabulary** (`aitd/slop.py`) counts hits against lists that were built by measuring which words and phrases LLMs use far
more often than people do: slop-forensics (Paech, 2025, MIT; essays from ten 2025 models including GPT-4.1-nano, GPT-4o-mini,
gpt-oss-20b, Gemini 2.5 Flash-Lite, Gemma 3, Llama 4 Scout, Mistral Nemo, Qwen3-30B, Grok 4 Fast and GLM-4.6, versus
human writing) for single words, bigrams and trigrams, and antislop-sampler (Paech, 2025, Apache 2.0) for 2,500 phrases and
three construction regexes drawn from creative writing. Rates are per thousand words. On the bundled controls the essay
word list alone separates human from machine at AUC about 0.8; the lists are vendored in `third_party/` with their licences.

**Which tells are dated.** The literature is clear that the famous 2023-24 vocabulary (delve, tapestry, testament,
"it's important to note", "In conclusion", "As of my last knowledge update", emoji bullets, "Certainly!") has been trained
out of 2026 frontier models by 41 to 86% even while it keeps rising in published corpora written with older tools, and that
em dashes went from a strong GPT-4.1 / Claude 4.x signature (9 to 11 per thousand words) to near zero in GPT-5.1+, Claude
Opus 5.5 and Gemini 3.1. The surviving 2026 tells are constructions rather than words: corrective framing ("not just X but
Y", "it's not X, it's Y", "rather than merely"), importance flagging ("this matters", "why X matters"), reframing
transitions ("adds another layer", "looking ahead"), and syntax such as present-participial riders and nominalisations.
Each list in this tool carries its year in the report's test table, the AUC column shows what it is worth on the controls,
and the combined score weights it accordingly; a hit on a 2023 word is editorial evidence about register, not authorship.
Population-level markers (Kobak, the em-dash studies) are stated by their authors not to be per-document detectors, and the
report treats the lexical layer the same way.

**Supervised classifier** (`aitd/classifier.py`): `tabularisai/ai-text-detection`, a ModernBERT-base model (149M parameters,
MIT) fine-tuned in 2026 on about 11 million rows dominated by the 2024 RAID benchmark. Its own card reports AUROC 0.99 on
RAID; independent tests of RAID-trained encoders show they collapse on 2025-26 generators (true-positive rate 0.13 at 1%
false positives on Qwen3 text in one study), and here it misses one of the Claude controls outright. Text is NFKC-normalised
before scoring, which closes the zero-width and homoglyph attacks. It is one vote in the combined score, weighted like every
other measure by how well it separates the controls, never a verdict.

**Combined score and concern level.** For every directional measure with AUC at least 0.70 on the controls under the model
set in use, the report fits the human and machine control values (Gaussian, log scale where the measure is log-scaled) and
computes the file's log-likelihood ratio between the two, clipped to plus or minus 4. The combined score is the average of
those ratios weighted by each measure's AUC above chance. It is validated leave-one-out: every control's score is computed
with that control left out of the fits, which gives the ensemble's own AUC (0.93 to 0.95 on the 67 controls) and the cut-point
that best separates them. The concern level reads the file's score against those two distributions: *low* below the
cut-point and not above three quarters of the human controls; *low to moderate* below the cut-point but in the top quarter
of human scores; *moderate* above the cut-point but below the machine median; *high* above the machine median. It is a
heuristic over a small control set, stated as such in the report, and a human-side result is weaker evidence than a
machine-side one for the reasons in the next section.

**Perplexity and burstiness** measure how predictable a text is to a language model and how much that varies from sentence
to sentence. Machine-generated text tends to be uniformly predictable; human writing swings. The idea goes back to GLTR
(Gehrmann et al., 2019) and the GPT-2 output-detector work (Solaiman et al., 2019), and underlies commercial tools such as
GPTZero. This repo computes document perplexity, mean sentence perplexity and the standard deviation across sentences under
`Qwen/Qwen2.5-1.5B` and `openai-community/gpt2`.

**Writing patterns** come from the detector engine of [avoid-ai-writing](https://github.com/conorbronsdon/avoid-ai-writing)
by Conor Bronsdon (MIT), vendored unmodified in `third_party/avoid-ai-writing/` and run in-process on the V8 engine that
ships with the `mini-racer` Python package. It applies several dozen deterministic rules: tiered vocabulary lists, transition
and template phrases, chatbot artifacts, generic conclusions, hedge stacking, formatting and rhythm signals. Refresh it with
`./scripts/update-style-rules.sh`.

**Hidden characters** are found with the Unicode character database that ships with Python: zero-width and format
characters, non-standard spaces, bidirectional controls, variation selectors, tag characters and private-use code points,
plus Latin words containing Cyrillic, Greek or Armenian lookalikes. These can carry watermarks or disguise text, but also
arrive through ordinary copy-and-paste; they say something about the file, not about who wrote the words.

## Choosing models

**Policy (October 2026): the default stays on current models.** New users will not install 2023-era Falcon or Qwen2.5
weights, so the default profile uses current open models even where an older pair scores better on one statistic, and the
report compensates by ranking tests by their measured reliability and leaning the combined score on the measures that hold
up. When a newer fully open base/instruct pair appears it should be baked off against the controls (one `profiles.json` entry
and one control pass; see the bake-off table below) and promoted if it wins. Candidates tried so far: Falcon3-7B (2024),
SmolLM3-3B (2025), OLMo-3-7B (2025-26); gated families (Gemma, Llama) are excluded because the installer must work without
a Hugging Face account.

Binoculars' published thresholds were derived for the Falcon-7B pair; the paper shows the method works with other
observer/performer pairs, but each pair needs its own thresholds, derived on labelled human and machine text. A newer pair
does not automatically separate current LLM output better: the all-Qwen3.5 `qwen35` profile is near chance on the controls
while OLMo-3 and SmolLM3 are not (see the bake-off below). Without recalibration the published 0.90 / 0.85 lines mean
nothing for any current pair, which is why every current profile reports positions among the controls and a concern level
instead of a verdict. With enough labelled text of your own you can derive thresholds for any pair and add them to
`profiles.json`.

For perplexity, a stronger model lowers every text's perplexity; what matters is the gap between the file and the controls,
which the reports show side by side.

### Measured: pair bake-off on the 67 controls (October 2026, Apple M5 Max)

AUC of each pair-based score alone (human direction) with the same Qwen3.5 small models for everything else:

| Observer / performer pair | Year | Binoculars | Fast-DetectGPT | Alignment imprint | Download |
| --- | --- | ---: | ---: | ---: | ---: |
| **OLMo-3-7B base / instruct** (default) | 2025-26 | **0.83** | 0.76 | 0.83 | 29 GB |
| SmolLM3-3B base / instruct (light) | 2025 | 0.81 | **0.79** | **0.90** | 12 GB |
| Falcon-7B / Falcon-7B-Instruct (the paper's pair) | 2023 | 0.80 | 0.79 | 0.79 | 29 GB |
| Qwen3.5-9B-Base / Qwen3.5-9B | 2026 | 0.62 | 0.60 | 0.59 | 39 GB |
| Falcon3-7B base / instruct | 2024 | 0.60 | 0.58 | 0.80 | 30 GB |

The single-model token statistics (surprisal spread 0.96, entropy 0.94, log-rank ratio 0.93, perplexity 0.92 from
Qwen3.5-2B-Base) and the classifier (0.91) do not depend on the pair and stay the strongest measures under every profile.
Two current open pairs match or beat the 2023 reference pair while the two other current pairs are near chance, so the
pair has to be chosen empirically, not by model age. New fully open pairs should be baked off the same way (one
`profiles.json` entry, one control pass) and promoted when they win.

## Why machine text can score "human", and which measures to trust

The verdict-style detectors (Binoculars, Fast-DetectGPT) are probability-curvature tests: they ask whether the text is
*more* predictable to a model than the alternatives that model would itself have written. That works when the text was
sampled from something close to the scoring model. It fails, in both directions, for several reasons that show up directly
in this repo's controls:

- **The generator is not the scorer.** Claude Fable 5.1 text scored by Falcon-7B (2023) or by a Qwen3.5 base model is text
  from a stronger, differently trained model with its own post-training style. It is not especially predictable to the
  scorer, so its curvature is small and its perplexity ordinary. The ChatGPT (2023) controls, close in era and training data
  to Falcon, score as machine; the Claude (2026) controls drift into the human range. The method detects "text that follows
  my model's preferences", not "text written by any machine". DetectRL measured the same thing: Binoculars AUROC 97 on
  Llama-2-70B text, 88 on ChatGPT, 55 on Claude-instant ([arXiv:2410.23746](https://arxiv.org/abs/2410.23746)); on GPT-4o
  summaries its true-positive rate at 1% false positives is 0.05 ([arXiv:2412.05139](https://arxiv.org/abs/2412.05139)).
- **What detectors learned is the instruction-tuning signature, not machine-ness.** Continuations from a *base* Llama-3-8B
  are rated human 97 to 99% of the time by GPTZero and Pangram, while the instruct model's continuations from the same
  weights are rated human 17 to 30% ([arXiv:2605.19516](https://arxiv.org/abs/2605.19516)); RLHF makes text *more*
  detectable ([arXiv:2503.17965](https://arxiv.org/abs/2503.17965)). Heavily prompted, style-guided output (a business
  email, a technical explainer) carries fewer of those artifacts and reads as human. The stylometry panels show the
  residue: longer words and higher lexical diversity for the machine controls, which to a base model looks like careful
  human prose.
- **Binoculars normalises away part of the signal when the pair agrees too much.** The score is performer perplexity divided
  by observer-performer cross-perplexity. If the observer and performer make nearly the same predictions (a 2026 base model
  and its own instruct version, trained on overlapping synthetic data), cross-perplexity tracks perplexity and the ratio
  compresses toward 1 for every text, human or machine. That is what the `qwen35` profile measures on the 67 controls: the
  surprisal statistics of Qwen3.5-2B-Base separate them with AUC 0.92 to 0.96 while the Qwen3.5-9B Binoculars ratio
  reaches only 0.62 and Fast-DetectGPT 0.60, close to chance. Independent
  measurements agree that the pair matters more than the statistic: Binoculars on a Qwen3-4B pair scored at chance (0.51)
  where a Gemma pair scored 0.93 ([arXiv:2610.15870](https://arxiv.org/abs/2610.15870)), and the Fast-DetectGPT authors
  report a Llama-3-8B base/instruct pair "substantially" beating the Falcon pair (repository README, January 2026). The
  alignment-imprint statistic, which keeps the base/instruct difference instead of dividing it away, recovers 8 to 14
  AUROC points on the same pair ([arXiv:2604.16923](https://arxiv.org/abs/2604.16923)).
- **Register, length and editing.** Short texts, lists, and formal genres sit in the overlap zone for every
  probability-based method; the paper-era thresholds were set on news and essays of a few hundred tokens. Human text
  that a model has rewritten is the hardest case of all: Binoculars' recall drops from 93.5% on direct LLM output to 15.1%
  on LLM-rewritten human text ([arXiv:2607.29539](https://arxiv.org/abs/2607.29539)).

What the 67 controls say about each measure (AUC of the measure alone, human direction; from the index page of each run,
October 2026):

| Measure | `original` / `--older-models` (Falcon pair, Qwen2.5-1.5B) | `qwen35` (Qwen3.5-9B pair, Qwen3.5-2B-Base) |
| --- | ---: | ---: |
| Surprisal spread (sd of token log-probability) | 0.89 | 0.96 |
| Mean next-token entropy | 0.91 | 0.94 |
| DetectLLM log-likelihood/log-rank ratio | 0.86 | 0.93 |
| Surprisal volatility | 0.81 | 0.93 |
| Mean log-rank of the actual token | 0.86 | 0.93 |
| Perplexity of the first small model | 0.85 | 0.92 |
| Perplexity shift under sentence shuffling | 0.88 | 0.91 |
| GLTR top-1 share / top-10 share | 0.81 / 0.83 | 0.91 / 0.90 |
| Sentence-length variation | 0.81 | 0.81 |
| Binoculars | 0.80 | 0.62 |
| Fast-DetectGPT (analytic) | 0.79 | 0.60 |
| Alignment imprint (unweighted) | 0.79 | 0.59 |
| Paragraph-length variation | 0.56 | 0.56 |
| AI-ism score | 0.70 | 0.70 |
| Lexical diversity (MATTR, MTLD, hapax) | 0.27 to 0.34 | 0.27 to 0.34 |
| Em dashes per 1,000 words | 0.50 | 0.50 |

Three practical conclusions. First, on this control set the cheapest signals (one forward pass of a 2B base model: surprisal
spread, entropy, log-rank, perplexity) beat the pair-based detectors by a wide margin, so the report leads with the measures
that separate the controls under the profile you ran rather than with a fixed favourite, and labels Binoculars and
Fast-DetectGPT as weak measures when they are. The texts that defeat the pair detectors come from both ends: polished 2026
Claude, o3 and GPT-5 essays read as human because they are not predictable to the scorer, and 2023 open-model abstracts
(Mistral-7B, MPT-30B, Llama-2-70B, GPT-2 XL, sampled at temperature) read as human because they are *too* unpredictable. Second, the "machine side" of any measure is stronger evidence than
the "human side", because every failure mode above pushes machine text toward the human range and almost nothing pushes
human text toward the machine range except brevity and formula. Third, the punctuation and vocabulary tells (em dashes,
"delve", rule-of-three lists) are near chance on these controls: Claude Fable 5.1 output in October 2026 has almost none of
them, and some human writers have many. They remain useful for editing and for older generators.

## Further methods, and what was not added

A survey of 2025-2026 work (October 2026) ranked these as the most useful additions for a local, training-free toolkit.
The first three are implemented here in reduced form; the rest are open.

| Method | Status | Notes |
| --- | --- | --- |
| Alignment imprint / LAPD ([arXiv:2604.16923](https://arxiv.org/abs/2604.16923)) | added, unweighted | log p(performer) − log p(observer) per token; the paper's information weighting and standardisation are not reproduced, and without them it is weak here (AUC 0.60 to 0.75) |
| Surprisal time-series features: Lastde ([arXiv:2410.06072](https://arxiv.org/abs/2410.06072)), SpecDetect ([arXiv:2508.11343](https://arxiv.org/abs/2508.11343)), DivEye ([arXiv:2509.18880](https://arxiv.org/abs/2509.18880)), late-stage stability ([arXiv:2601.04833](https://arxiv.org/abs/2601.04833)) | added, four simple statistics | spread, volatility, lag-1 autocorrelation, second-half/first-half volatility of the log-probability sequence; the papers' full estimators (diversity entropy, DFT energy, nine-moment set) are not |
| Perplexity under sentence shuffling, Luminol-AIDetect ([arXiv:2604.25860](https://arxiv.org/abs/2604.25860)) | added as one log ratio | the paper uses a distribution over many shuffles and density estimation; on these controls machine text loses more predictability when shuffled (AUC 0.81 to 0.83) |
| Telescope ([arXiv:2607.04061](https://arxiv.org/abs/2607.04061)): token-repetition probability from a tiny reference model; reported to beat Binoculars on polished essays | not added | definition needs the full paper; one extra small-model pass |
| A second observer/performer pair chosen on the controls (Gemma or Llama-3 base/instruct), MOSAIC-style ensembling ([arXiv:2409.07615](https://arxiv.org/abs/2409.07615)) | not added | needs ~16 GB more models; the evidence above says the pair choice is the single biggest lever for Binoculars and Fast-DetectGPT |
| Conformal thresholds from the human controls; report TPR at 1% FPR ([arXiv:2505.05084](https://arxiv.org/abs/2505.05084)) | not added | 35 human controls is still thin for a 1% false-positive bound |
| One supervised vote: `tabularisai/ai-text-detection` (ModernBERT, 150M, MIT) | not added | strong in-distribution, collapses on 2025-26 generators in independent tests (TPR at 1% FPR 0.13 on Qwen3 text); would need a capped weight |
| Larger, current calibration sets: `mattc95/2026-AI-DETECTOR-BENCHMARK` (MIT, 14 models, Feb 2026), `textsightai/ai-text-detection-benchmark-2026` (CC BY-SA) | partly done | one text per model from the 2026 benchmark is in the controls; the full 500-text set would lift the control count to hundreds |
| Adversarial harness: `raid-bench`, PADBen, released adversarial paraphrases | not added | to measure how the measures degrade under paraphrase and humanizer tools |
| Data-derived vocabulary lists (antislop-sampler, slop-forensics) and a pure-Python pattern scorer (sloptrim) beside avoid-ai-writing | not added | the AI-ism rules scored 0.50 AUC on these controls, so better lists are the fix |
| Hidden-character coverage audit against demarkify (Hangul fillers, tag characters, Cf/Co catch-all) and `confusable_homoglyphs` | not added | current check covers zero-width, bidi, variation selectors, tags, Cf/Co and Cyrillic/Greek/Armenian lookalikes |

Not recommended by the survey: intrinsic-dimension (PHD) scores (near-zero recall at 1% FPR), prompting an LLM to judge
(below GPTZero even with frontier models), regeneration methods such as DNA-GPT (cost), and any single vocabulary rule as
decisive evidence. Statistical watermarks (SynthID-Text, KGW) are only detectable with the generator's key, so no blind
watermark check is possible; hidden-character scanning is the only watermark-adjacent signal available.

StoryScope ([arXiv:2604.03136](https://arxiv.org/abs/2604.03136)) separates AI fiction from human fiction by narrative
features (character agency, chronology, escalation, moral ambiguity) rather than style, at 93% macro-F1 on 5,000-word
stories. The features are extracted by an LLM, so it is not a local zero-shot method and is not added; its figure style
(a distribution per source with the sample marked against it) is what the violin panels in this report follow.

## Watermarks and provenance: what can and cannot be known locally

Verified as of October 2026 (primary sources listed in References):

- **Claude carries a statistical watermark, and it is undetectable here.** Anthropic announced in August 2026 that models
  released from 2026-08-02 (the Claude 5 family, Opus and Sonnet 4.5 onward, Haiku 4.5 onward, with older models being
  back-filled) carry a SynthID-Text-style watermark: the sampler's source of randomness is keyed, which biases word choice
  statistically. In Anthropic's words, "nothing is added to the text and there are no hidden characters." Detection is a
  private-preview API for regulators, researchers, educators and similar parties; there is no public checker, light edits
  probably survive, a full rewrite removes it, and it is weak on short samples. Keyless detection methods published so far
  need hundreds of live queries to the generating model and are not validated on production Claude, so a static-text tool
  cannot test for it. Claims that Claude inserts zero-width or no-break characters are contradicted by Anthropic's statement
  and by independent clipboard tests; the one real Claude-specific artifact is an HTML wrapper class
  (`font-claude-response-body`) that survives select-and-copy into rich-text editors, which this tool now flags when it
  appears in the input and which proves only that the text passed through the Claude interface.
- **OpenAI**: the narrow no-break space (U+202F) episode was o3 and o4-mini in April 2025 (OpenAI called it a
  reinforcement-learning quirk, not a watermark) and GPT-5 non-reasoning in October 2025; no verified 2026 recurrence.
  OpenAI's own statistical watermark ("textGrain") is EU-only and off by default on the API, with no public detector.
- **Google**: SynthID-Text is applied in the Gemini app; the public SynthID portal covers images, video and audio only.

So the hidden-character check is a **pipeline** classifier: no-break spaces after sentence ends, curly quotes and the
ellipsis character point at a word processor, a web page or a chat window as the copy source; ligatures and soft hyphens
point at a PDF; narrow no-break spaces point at 2025-era OpenAI output. None of it identifies the author.

## Limits

No detector proves authorship. Edited or paraphrased machine text can pass. Short texts, formulaic genres, lists, and text
dense with names and numbers can score as machine-like when written by a person. The model-based checks are weaker on text
the models have memorised (famous passages) and on languages other than English; the pattern rules are English-only. Use the
scores alongside drafts, sources and the writer's history, never as a verdict on their own.

## References

- Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping,
  Tom Goldstein. **Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text.** ICML 2024.
  [arXiv:2401.12070](https://arxiv.org/abs/2401.12070). Reference code: [ahans30/Binoculars](https://github.com/ahans30/Binoculars)
  (used as the specification for the score; not copied).
- Junxi Wu, Kailin Huang, Dongjian Hu, Bin Chen, Hao Wu, Shu-Tao Xia, Changliang Zou. **Alignment Imprint: Zero-Shot
  AI-Generated Text Detection via Provable Preference Discrepancy.** 2026. [arXiv:2604.16923](https://arxiv.org/abs/2604.16923).
  Source of the alignment-imprint (LAPD) idea.
- Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, Fei Wu. **Training-free LLM-generated Text
  Detection by Mining Token Probability Sequences.** 2024. [arXiv:2410.06072](https://arxiv.org/abs/2410.06072). Lastde.
- Advik Raj Basani, Pin-Yu Chen. **Diversity Boosts AI-Generated Text Detection.** TMLR 2026.
  [arXiv:2509.18880](https://arxiv.org/abs/2509.18880). DivEye.
- Haitong Luo, Weiyao Zhang, Suhang Wang, Wenji Zou, Chungang Lin, Xuying Meng, Yujun Zhang. **SpecDetect: Simple, Fast,
  and Training-Free Detection of LLM-Generated Text via Spectral Analysis.** AAAI 2026.
  [arXiv:2508.11343](https://arxiv.org/abs/2508.11343).
- Ke Sun, Guangsheng Bao, Han Cui, Yue Zhang. **When AI Settles Down: Late-Stage Stability as a Signature of AI-Generated
  Text Detection.** 2026. [arXiv:2601.04833](https://arxiv.org/abs/2601.04833).
- Lucio La Cava, Andrea Tagarelli. **Luminol-AIDetect: Fast Zero-shot Machine-Generated Text Detection based on Perplexity
  under Text Shuffling.** 2026. [arXiv:2604.25860](https://arxiv.org/abs/2604.25860).
- Yixuan Even Xu, Ziqian Zhong, Aditi Raghunathan, Fei Fang, J. Zico Kolter. **Base Models Look Human To AI Detectors.**
  2026. [arXiv:2605.19516](https://arxiv.org/abs/2605.19516).
- Jenna Russell, Rishanth Rajendhran, Chau Minh Pham, Mohit Iyyer, John Wieting. **StoryScope: Investigating idiosyncrasies
  in AI fiction.** 2026. [arXiv:2604.03136](https://arxiv.org/abs/2604.03136).
- Anthropic. **How Claude's text watermark works.** 2026-08-14, updated 2026-09-01. anthropic.com/news/claude-text-watermark;
  and support article "How Claude marks AI-generated content" (support.claude.com, article 16266773).
- Rumi. **ChatGPT's o3 and o4-mini insert narrow no-break spaces.** 2025-04-20, with OpenAI's 2025-04-22 response (rumidocs.com).
- Thibaud Gloaguen, Nikola Jovanović, Robin Staab, Martin Vechev. **Black-Box Detection of Language Model Watermarks.**
  ICLR 2025. [arXiv:2405.20777](https://arxiv.org/abs/2405.20777). Keyless detection needs live model queries.
- Sumanth Dathathri et al. **Scalable watermarking for identifying large language model outputs (SynthID-Text).** Nature, 2024.
- Mason Freeburg. **Em-dash usage across language models.** 2026. [arXiv:2603.27006](https://arxiv.org/abs/2603.27006).
- Dmitry Kobak, Rita González-Márquez, Emőke-Ágnes Horvát, Jan Lause. **Delving into LLM-assisted writing in biomedical
  publications through excess vocabulary.** Science Advances 11(27), 2025. [arXiv:2406.07016](https://arxiv.org/abs/2406.07016).
  Word list: github.com/berenslab/llm-excess-vocab (MIT), vendored in `third_party/llm-excess-vocab/`.
- Tom S. Juzek, Zina B. Ward. **Why Does ChatGPT "Delve" So Much? Exploring the Sources of Lexical Overrepresentation in
  Large Language Models.** COLING 2025. [arXiv:2412.11385](https://arxiv.org/abs/2412.11385).
- Alex Reinhart et al. **Do LLMs write like humans? Variation in grammatical and rhetorical styles.** PNAS, 2025.
  [arXiv:2410.16107](https://arxiv.org/abs/2410.16107).
- Weixin Liang et al. **Mapping the Increasing Use of LLMs in Scientific Papers.** 2024. [arXiv:2403.07183](https://arxiv.org/abs/2403.07183).
- WikiProject AI Cleanup. **Signs of AI writing.** Wikipedia, 2025-26, CC BY-SA 4.0.
- Graphite / Five Percent. **AI Tells** and **Opus 5.5 update**, 2026 (graphite.io/five-percent/research); cross-generation
  tell study; used as a reference for the 2026 construction frames, no data vendored.
- Sam Paech. **antislop-sampler** (Apache 2.0) and **slop-forensics** (MIT), 2025. Lists vendored in `third_party/`.
- Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, Yue Zhang. **Fast-DetectGPT: Efficient Zero-Shot Detection of
  Machine-Generated Text via Conditional Probability Curvature.** ICLR 2024. [arXiv:2310.05130](https://arxiv.org/abs/2310.05130).
  Reference code: [baoguangsheng/fast-detect-gpt](https://github.com/baoguangsheng/fast-detect-gpt) (used as the
  specification for the analytic score; not copied).
- Jinyan Su, Terry Yue Zhuo, Di Wang, Preslav Nakov. **DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of
  Machine-Generated Text.** 2023. [arXiv:2306.05540](https://arxiv.org/abs/2306.05540). Source of the LRR measure.
- Michael A. Covington, Joe D. McFall. **Cutting the Gordian Knot: The Moving-Average Type-Token Ratio (MATTR).** Journal of
  Quantitative Linguistics 17(2), 2010.
- Philip M. McCarthy, Scott Jarvis. **MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical
  diversity assessment.** Behavior Research Methods 42, 2010.
- Sebastian Gehrmann, Hendrik Strobelt, Alexander M. Rush. **GLTR: Statistical Detection and Visualization of Generated
  Text.** ACL 2019 (system demonstrations). [arXiv:1906.04043](https://arxiv.org/abs/1906.04043).
- Irene Solaiman et al. **Release Strategies and the Social Impacts of Language Models.** 2019.
  [arXiv:1908.09203](https://arxiv.org/abs/1908.09203). Introduced perplexity-based and classifier-based GPT-2 output detection.
- Edward Tian, Alexander Cui. GPTZero (2023). Popularised perplexity plus burstiness as a detection heuristic; no paper.
- Biyang Guo et al. **How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection.** 2023.
  [arXiv:2301.07597](https://arxiv.org/abs/2301.07597). Source of the HC3 human and ChatGPT control texts (CC BY-SA 4.0).
- Conor Bronsdon. **avoid-ai-writing.** [github.com/conorbronsdon/avoid-ai-writing](https://github.com/conorbronsdon/avoid-ai-writing),
  MIT. The writing-pattern detector vendored in `third_party/`.
- Ebtesam Almazrouei et al. **The Falcon Series of Open Language Models.** 2023.
  [arXiv:2311.16867](https://arxiv.org/abs/2311.16867). Models: `tiiuae/falcon-7b`, `tiiuae/falcon-7b-instruct` (Apache 2.0).
- Qwen Team. **Qwen2.5 Technical Report.** 2024. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).
  Models: `Qwen/Qwen2.5-1.5B` and the Qwen3.5 family (Apache 2.0), including the small base models used by every current profile.
- Alec Radford et al. **Language Models are Unsupervised Multitask Learners.** 2019. Model: `openai-community/gpt2` (MIT).

## Controls

35 human texts (public-domain books, essays and letters from 1532 to 1919 across fiction, science, philosophy, memoir and
economics; Reddit, finance, medical and open-domain Q&A answers from the HC3 corpus; arXiv abstracts from RAID) and 32
machine texts (ChatGPT answers from January 2023; Claude Fable 5.1 output from October 2026; one text each from Claude 3.5,
3.7, Sonnet 4 and 4.5, GPT-3.5, GPT-4o, GPT-4.1, GPT-5, GPT-5 mini, o3, DeepSeek V3, Grok 4 and Kimi K2 via the MIT-licensed
2026 AI Detector Benchmark; Mistral-7B-Instruct, MPT-30B-Chat, Llama-2-70B-Chat and GPT-2 XL from RAID; three texts
generated locally by Qwen3.5-9B). Sources, licences and word counts are in [`controls/README.md`](controls/README.md). Drop your own files
into `controls/human/` or `controls/ai/` and they are included automatically.

## License

MIT for this repo's code. `third_party/avoid-ai-writing/` is MIT, Copyright Conor Bronsdon. Model weights are downloaded
separately under their own licenses.
