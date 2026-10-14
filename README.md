# ai-text-detector

Is this text machine-generated? Run three independent, fully local checks and get a printable report per file.

1. **Binoculars** (Hans et al., ICML 2024): the best-regarded zero-shot detector, using the paper's exact Falcon-7B model pair
   and thresholds.
2. **Perplexity and burstiness** (the GLTR / GPTZero family) under two language models.
3. **Writing-pattern and hidden-character checks**: AI-isms such as "delve", "in today's fast-paced world" and
   "in conclusion" (rules from the avoid-ai-writing project), plus zero-width characters, bidi controls and lookalike letters
   from other alphabets.

No API keys, no uploads, nothing leaves your machine. The installer assumes nothing: no Homebrew, no Python, no Node.

## Quick start

```sh
git clone https://github.com/myster-t/ai-text-detector.git
cd ai-text-detector
./install.sh            # installs a Python runtime, the packages and ~33 GB of models; ends with a self-test
./detect essay.txt      # writes reports/essay-report.html and reports/index-report.html
./detect essay.txt --profile latest   # optional: newer models, no thresholds (see Model profiles)
```

`./detect` takes any number of text files and an optional `--out DIR`. The models load for the run and are released when it
exits; nothing stays resident. Everything lives inside the checkout (`.venv/`, `models/`), so deleting the folder removes it all.

Requirements: macOS or Linux (Windows via WSL2) and a machine with 32 GB or more of memory for the 7B to 9B model pairs
(Apple Silicon or an NVIDIA GPU; CPU works but is slow). A run on an Apple M5 Max takes two to three minutes, most of it
loading weights.

### Disk space

The models are the cost. They are downloaded once into `./models` (or wherever `--models-dir` points) and never duplicated.

| What | Size on disk |
| --- | ---: |
| Python environment (`.venv`, PyTorch and Transformers) | about 3 GB |
| `original` profile: Falcon-7B, Falcon-7B-Instruct, Qwen2.5-1.5B, GPT-2 | 33 GB |
| `latest` profile: Qwen3.5-9B-Base, Qwen3.5-9B, Qwen3.5-2B-Base, Qwen3.5-0.8B-Base | 45 GB |
| Both profiles | 78 GB |

The installer checks free space at the models location before downloading. To put the models on another disk:

```sh
./install.sh --models-dir /Volumes/External/aitd-models      # remembered in .models-dir for every later ./detect
```

`AITD_MODELS_DIR` in the environment overrides that file. To remove what was installed:

```sh
./uninstall.sh --profile latest     # just that profile's models (shared models are kept)
./uninstall.sh --all-models         # every model
./uninstall.sh --all                # models and the Python environment
./uninstall.sh --profile latest --dry-run   # preview what a removal would free
```

Nothing is installed outside the checkout except uv itself (in `~/.local/bin`), which other tools may use; remove it with
`rm ~/.local/bin/uv ~/.local/bin/uvx` if you want it gone too.

## Expected results

Every run re-scores 21 bundled controls (9 human, 12 machine; see [`controls/README.md`](controls/README.md)) with the same
models as your file, and every report draws your file on a number line among them. The two anchor controls score like this
under the default profile:

| Text | Binoculars | Verdict | Qwen2.5-1.5B perplexity | Burstiness | GPT-2 perplexity | AI-ism score |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `controls/human/stewart-letter-1913.txt` (1913 letter, public domain) | 0.972 | human | 24.8 | 43.8 | 47.5 | 1 / 100 |
| `controls/ai/claude-marketing-paragraph.txt` (unedited LLM output) | 0.800 | AI | 4.9 | 10.0 | 15.5 | 12 / 100 |

The index page shows the full control ranges for the profile you ran. The installer's self-test checks these two.

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

`profiles.json` defines two model sets. `./detect` and `./install.sh` take `--profile NAME`.

| Profile | Binoculars pair | Perplexity models | Thresholds |
| --- | --- | --- | --- |
| `original` (default) | tiiuae/falcon-7b, falcon-7b-instruct | Qwen2.5-1.5B, GPT-2 | the paper's; reports state a verdict |
| `latest` | Qwen/Qwen3.5-9B-Base, Qwen3.5-9B | Qwen3.5-2B-Base, Qwen3.5-0.8B-Base | none; reports give no verdict |

Binoculars' published thresholds were derived for the Falcon pair and mean nothing for any other, so the `latest` profile
deliberately draws no lines and makes no call. Its reports show where your file falls on each measure relative to the 21
controls, as dots on a number line, and say which side of the gap between the human and machine groups it lands on.

```sh
./install.sh --profile latest                 # downloads the latest models (45 GB); original stays if present
./detect --profile latest essay.txt
```

Profiles are plain entries in `profiles.json` (observer, performer, perplexity models, optional thresholds). Add one to try
any other pair; the installer, runner and uninstaller all read the same file, so a new profile needs no code changes.

## What a report contains

- The Binoculars score and, under the `original` profile, a verdict against both published thresholds.
- Number-line strips for Binoculars, perplexity and burstiness showing the human and machine controls as dots and your file
  as a marker, with the threshold lines drawn only when they apply.
- The file's numbers next to every control, re-scored in the same run.
- A sentence-by-sentence perplexity table with bars, showing where a text is unusually predictable.
- The writing-pattern findings: each flagged phrase, its category, severity and the suggested plainer alternative.
- Hidden or non-standard characters and mixed-script words, with positions.
- A methods and limits section.

Reports are self-contained HTML (no scripts, fonts or external assets), print to A4, and render in light and dark mode.

## Building blocks

Each check can run on its own and prints JSON:

```sh
.venv/bin/python binoculars.py draft.txt
.venv/bin/python perplexity.py Qwen/Qwen2.5-1.5B draft.txt
.venv/bin/python style.py draft.txt
.venv/bin/python unicode_check.py draft.txt
```

The scripts find the models in `./models` and run offline by default; no environment variables are needed.

## How it works

**Binoculars** scores a text as the ratio of its perplexity under a "performer" model to the cross-perplexity between an
"observer" model and the performer. The cross-perplexity measures how surprising the text is to one language model reading
another's predictions, which normalises away prompts and topics that make text predictable for reasons other than being
machine-written. Texts that a model finds much more predictable than this baseline score low. This repo uses the paper's
reference pairing, `tiiuae/falcon-7b` as observer and `tiiuae/falcon-7b-instruct` as performer, and its published thresholds.
`binoculars.py` is about 40 lines and follows Section 3 of the paper.

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

### Measured: original vs latest on the 21 controls (October 2026, Apple M5 Max)

| | `original` (Falcon-7B pair, 2023) | `latest` (Qwen3.5-9B pair, 2026) |
| --- | --- | --- |
| Human controls, Binoculars range (mean) | 0.900 to 1.010 (0.964) | 0.898 to 1.098 (0.991) |
| Machine controls, Binoculars range (mean) | 0.513 to 0.993 (0.777) | 0.754 to 1.040 (0.918) |
| Gap between group means | 0.187 | 0.073 |
| Machine controls scoring inside the human range | 3 of 12 (all Claude 2026: business email, personal essay, technical explainer) | 7 of 12 (those three plus the Claude marketing paragraph and three ChatGPT answers) |

Newer is not better here. The 2026 pair leaves less room between the groups and lets more machine text through, which
matches a known weakness of the method: as base models train on more synthetic text, machine prose stops surprising them.
`latest` stays in the repo as a working example of a second profile and of the uncalibrated report, not as a recommendation.
The practical lesson from both profiles is the same: polished Claude output in formal registers (an email, a technical
explanation, a personal essay) scores inside the human range for Binoculars, so a human-side reading is weaker evidence than a
machine-side one.

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

Nine human texts (public-domain letters and memoirs from 1879 to 1914, and Reddit, finance and medical Q&A answers from the
HC3 corpus) and twelve machine texts (ChatGPT answers from January 2023 and Claude Fable 5.1 output from October 2026 across
five registers). Sources, licences and word counts are in [`controls/README.md`](controls/README.md). Drop your own files
into `controls/human/` or `controls/ai/` and they are included automatically.

## License

MIT for this repo's code. `third_party/avoid-ai-writing/` is MIT, Copyright Conor Bronsdon. Model weights are downloaded
separately under their own licenses.
