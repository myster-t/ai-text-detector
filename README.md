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
6. **Writing-pattern and hidden-character checks**: AI-isms such as "delve", "in today's fast-paced world" and
   "in conclusion" (rules from the avoid-ai-writing project), plus zero-width characters, bidi controls and lookalike letters
   from other alphabets, each marked inline in the text.

No API keys, no uploads, nothing leaves your machine. The installer assumes nothing: no Homebrew, no Python, no Node.

## Quick start

```sh
git clone https://github.com/myster-t/ai-text-detector.git
cd ai-text-detector
./install.sh                    # installs a Python runtime, the packages and ~45 GB of current models; ends with a self-test
./detect essay.txt              # writes reports/essay-report.html and reports/index-report.html
./install.sh --older-models     # optional: the 2023 Falcon-7B / Qwen2.5 set with the Binoculars paper's thresholds (~33 GB)
./detect essay.txt --older-models
```

The default is the current-generation model set (Qwen3.5, 2026). It has no published Binoculars thresholds, so the report
never says "AI" or "human" outright; it shows where the text falls on about thirty measures against 67 control texts and
which of those measures separate the controls. `--older-models` switches every script to the original Falcon-7B pair
with the paper's thresholds, which adds a Binoculars verdict line. Both sets can be installed side by side.

`./detect` takes any number of text files and an optional `--out DIR`. The models load for the run and are released when it
exits; nothing stays resident. The control texts are scored once per model set and cached in `controls/scores/<profile>.json`
(committed to the repo, keyed by each control's text hash and the profile's model names), so a normal run scores only your
files; a control is re-scored only when its text or the model set changes, or when you pass `--rescore-controls`. Everything lives inside the checkout (`.venv/`, `models/`), so deleting the folder removes it all.

Requirements: macOS or Linux (Windows via WSL2) and a machine with 32 GB or more of memory for the 7B to 9B model pairs
(Apple Silicon or an NVIDIA GPU; CPU works but is slow). A run on an Apple M5 Max takes two to three minutes, most of it
loading weights.

### Disk space

The models are the cost. They are downloaded once into `./models` (or wherever `--models-dir` points) and never duplicated.

| What | Size on disk |
| --- | ---: |
| Python environment (`.venv`, PyTorch and Transformers) | about 3 GB |
| default set (`latest`): Qwen3.5-9B-Base, Qwen3.5-9B, Qwen3.5-2B-Base, Qwen3.5-0.8B-Base | 45 GB |
| `--older-models` set (`original`): Falcon-7B, Falcon-7B-Instruct, Qwen2.5-1.5B, GPT-2 | 33 GB |
| Both profiles | 78 GB |

The installer checks free space at the models location before downloading. To put the models on another disk:

```sh
./install.sh --models-dir /Volumes/External/aitd-models      # remembered in .models-dir for every later ./detect
```

`AITD_MODELS_DIR` in the environment overrides that file. To remove what was installed:

```sh
./uninstall.sh --profile original   # just the --older-models set (shared models are kept); --profile latest = the default set
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

Under the default set the same two controls score Binoculars 1.016 (human) and 0.879 (machine), with no verdict line; the
self-test checks that the human anchor scores above the machine anchor and that the group means separate. The index page of
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

The rendered reports for both controls are in [`examples/`](examples/).

## Model profiles

`profiles.json` defines the model sets. `./detect`, `./install.sh` and `./uninstall.sh` take `--profile NAME`;
`--older-models` is shorthand for `--profile original`, and no flag means `latest`.

| Profile | Binoculars pair | Perplexity models | Thresholds |
| --- | --- | --- | --- |
| `latest` (default) | Qwen/Qwen3.5-9B-Base, Qwen3.5-9B | Qwen3.5-2B-Base, Qwen3.5-0.8B-Base | none; reports give no verdict |
| `original` (`--older-models`) | tiiuae/falcon-7b, falcon-7b-instruct | Qwen2.5-1.5B, GPT-2 | the paper's; reports state a verdict |
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

- A verdict box: the Binoculars score and, under `--older-models`, a verdict against both published thresholds; plus a
  count of how many directional measures put the file on the human or machine side of the control gap, overall and among
  the measures that separate the controls well under the profile you ran.
- Summary tiles (Binoculars, Fast-DetectGPT, perplexity, GLTR top-10 share, AI-isms, sentence-length variation, hidden
  characters), each with a bar showing the human and machine control ranges and the file's position.
- **The shape of the text**: a violin panel for every measure (about 30), human controls in green and machine controls in
  red with the individual control points and medians, the cut-point that best separates the two groups as a dotted line,
  and the file as a line across both. The axis always runs with the human direction up; the file line and the panel border
  are green or red for the side of the cut the file lands on; each panel states the AUC with which that measure alone
  separates the controls, and panels under 0.7 are drawn grey, dashed and desaturated as weak evidence, so the red panels
  are the ones to read.
- Number-line strips for Binoculars and Fast-DetectGPT with threshold lines drawn only when they apply.
- A **token heat map**: the text with every token coloured by where the real next token sat in the model's ranking
  (top-1, top-10, top-100, top-1000, beyond); hover for rank, log-probability and entropy.
- The **annotated text**: every AI-ism match marked inline by severity, sentence spans the pattern detector flags, each
  hidden or non-standard character shown as a red code-point badge in place, and mixed-script words outlined. The pattern
  and hidden-character tables below it give positions and context.
- A sentence-by-sentence perplexity table with bars, showing where a text is unusually predictable.
- Every measurement for the file and every control in a collapsed table (click to open), and a methods and limits section.
- The index page lists all files, the control table, and a ranking of all measures by how well they separate the controls.

Reports are self-contained HTML (no scripts, fonts or external assets), print to A4, and render in light and dark mode.

## Building blocks

Each check can run on its own and prints JSON:

```sh
.venv/bin/python binoculars.py draft.txt
.venv/bin/python perplexity.py Qwen/Qwen2.5-1.5B draft.txt
.venv/bin/python lmfeat.py Qwen/Qwen2.5-1.5B draft.txt     # GLTR buckets, log-rank, LRR, entropy
.venv/bin/python features.py draft.txt                      # stylometry, no model
.venv/bin/python style.py draft.txt
.venv/bin/python unicode_check.py draft.txt
```

`charts.py` holds the SVG builders (violins, strips, position bars, token heat map) and has no dependencies.

The scripts find the models in `./models` and run offline by default; no environment variables are needed.

## How it works

**Binoculars** scores a text as the ratio of its perplexity under a "performer" model to the cross-perplexity between an
"observer" model and the performer. The cross-perplexity measures how surprising the text is to one language model reading
another's predictions, which normalises away prompts and topics that make text predictable for reasons other than being
machine-written. Texts that a model finds much more predictable than this baseline score low. This repo uses the paper's
reference pairing, `tiiuae/falcon-7b` as observer and `tiiuae/falcon-7b-instruct` as performer, and its published thresholds.
`binoculars.py` is about 40 lines and follows Section 3 of the paper.

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

**Stylometry** (`features.py`) needs no model: moving-average type-token ratio (Covington and McFall, 2010), MTLD (McCarthy
and Jarvis, 2010), hapax ratio, sentence- and paragraph-length coefficient of variation, word length, and per-thousand-word
rates of em dashes, semicolons, colons, questions, exclamations, contractions, first-person pronouns, "not X, but Y"
constructions and three-item lists. These describe register and rhythm; they shift with genre and separate the controls
only partly, which the report shows honestly through each panel's AUC.

**Perplexity and burstiness** measure how predictable a text is to a language model and how much that varies from sentence
to sentence. Machine-generated text tends to be uniformly predictable; human writing swings. The idea goes back to GLTR
(Gehrmann et al., 2019) and the GPT-2 output-detector work (Solaiman et al., 2019), and underlies commercial tools such as
GPTZero. This repo computes document perplexity, mean sentence perplexity and the standard deviation across sentences under
`Qwen/Qwen2.5-1.5B` and `openai-community/gpt2`.

**Writing patterns** come from the detector engine of [avoid-ai-writing](https://github.com/conorbronsdon/avoid-ai-writing)
by Conor Bronsdon (MIT), vendored unmodified in `third_party/avoid-ai-writing/` and run in-process on the V8 engine that
ships with the `mini-racer` Python package. It applies several dozen deterministic rules: tiered vocabulary lists, transition
and template phrases, chatbot artifacts, generic conclusions, hedge stacking, formatting and rhythm signals. Refresh it with
`./update-style-rules.sh`.

**Hidden characters** are found with the Unicode character database that ships with Python: zero-width and format
characters, non-standard spaces, bidirectional controls, variation selectors, tag characters and private-use code points,
plus Latin words containing Cyrillic, Greek or Armenian lookalikes. These can carry watermarks or disguise text, but also
arrive through ordinary copy-and-paste; they say something about the file, not about who wrote the words.

## Choosing models

The defaults are deliberate. Binoculars' thresholds were derived for the Falcon-7B pair; the paper shows the method works
with other observer/performer pairs, but each pair needs its own thresholds, derived on labelled human and machine text. A
newer pair such as the Qwen3.5 models in the `latest` profile will likely separate current LLM output better, because newer
post-trained models are closer to what people actually use, but without recalibration the published 0.90 / 0.85 lines mean
nothing, which is why that profile reports positions among the controls instead of verdicts. With enough labelled text of
your own you can derive thresholds for any pair and add them to `profiles.json`.

For perplexity, a stronger model lowers every text's perplexity; what matters is the gap between the file and the controls,
which the reports show side by side.

### Measured: original vs latest on the 67 controls (October 2026, Apple M5 Max)

| | `original` (Falcon-7B pair, 2023) | `latest` (Qwen3.5-9B pair, 2026) |
| --- | --- | --- |
| Binoculars AUC on the 67 controls | 0.80 | 0.62 |
| Fast-DetectGPT AUC | 0.79 | 0.60 |
| Best single-model statistic (AUC) | entropy 0.91 | surprisal spread 0.96 |

Newer is not better for the pair-based scores: the 2026 pair leaves almost no room between the groups. It is better for the
single-model statistics, which are the strongest measures under either profile. The pair choice is a known lever (see the
next section), and candidate pairs can be tried by adding a profile; the Falcon3-7B pair is included as `falcon3` for that
purpose.

Even so, `latest` is the default since October 2026: the original pair's advantage is its calibrated verdict line, and
that line is only meaningful for 2023-era generators; for current text the report's value is the control comparison on
every measure, which the newer small base model does better (see the next section), and newer models keep arriving.
The practical lesson from both profiles is the same: polished Claude output in formal registers (an email, a technical
explanation, a personal essay) scores inside the human range for Binoculars, so a human-side reading is weaker evidence than a
machine-side one.

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
  compresses toward 1 for every text, human or machine. That is what the `latest` profile measures on the 67 controls: the
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

| Measure | `original` / `--older-models` (Falcon pair, Qwen2.5-1.5B) | `latest` / default (Qwen3.5-9B pair, Qwen3.5-2B-Base) |
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
  Models: `Qwen/Qwen2.5-1.5B` and, in the `latest` profile, the Qwen3.5 family (Apache 2.0).
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
