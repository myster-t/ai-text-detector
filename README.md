# ai-text-detector

Local, zero-shot detection of machine-generated text with a printable report. Two independent methods, both run entirely on
your machine with open models: **Binoculars** (Hans et al., ICML 2024) and **perplexity with burstiness** (the GLTR / GPTZero
family). No API keys, no uploads, no trained classifier to drift.

The scripts implement the published methods directly; nothing here is cloned from another repository.

## Expected results

Every run re-scores two bundled controls so each report carries its own reference points. Numbers below are from an Apple
M5 Max, October 2026, with the exact models and thresholds in this repo.

| Text | Binoculars | Verdict | Qwen2.5-1.5B perplexity | Burstiness | GPT-2 perplexity |
| --- | ---: | --- | ---: | ---: | ---: |
| `controls/human-letter-1913.txt` (1913 letter, public domain) | 0.972 | human | 24.8 | 43.8 | 47.5 |
| `controls/ai-marketing-paragraph.txt` (unedited LLM output) | 0.800 | AI | 4.9 | 10.0 | 15.5 |

Binoculars reads **below 0.9015** as machine-generated at the paper's accuracy-optimal setting and **below 0.8536** at its
low-false-positive setting; higher is more human-like. Machine text tends to sit near perplexity 5 with burstiness near 10;
human prose runs far higher on both. In testing on nine polished personal essays written by a student, Binoculars scored
1.04 to 1.28 and perplexity 24 to 107, well on the human side.

See [`examples/`](examples/) for the rendered reports of both controls, including the per-sentence breakdown.

## Install

Requires Python 3.11+, about 33 GB of disk for the models, and a GPU or Apple Silicon with 32 GB+ of unified memory for the
Falcon-7B pair (bf16, about 29 GB resident during a run). CPU works but is slow.

```sh
git clone https://github.com/myster-t/ai-text-detector.git
cd ai-text-detector
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# download the models once (Hugging Face CLI: https://huggingface.co/docs/huggingface_hub/guides/cli)
hf download tiiuae/falcon-7b          --include "*.safetensors" --include "*.json" --include "tokenizer*"
hf download tiiuae/falcon-7b-instruct --include "*.safetensors" --include "*.json" --include "tokenizer*"
hf download Qwen/Qwen2.5-1.5B
hf download openai-community/gpt2     --include "*.safetensors" --include "*.json" --include "*.txt"
```

## Run

```sh
HF_HUB_OFFLINE=1 .venv/bin/python report.py --out reports draft1.txt draft2.txt
```

Writes `reports/<name>-report.html` for each input and `reports/index-report.html` with a summary table. Reports are
self-contained HTML (no scripts, fonts or external assets), print to A4, and work in light and dark mode. Models are loaded
for the run and released when it exits; nothing stays resident.

Each report contains:

- the Binoculars score and verdict against both published thresholds
- the file's numbers next to the two controls, re-scored in the same run
- a sentence-by-sentence perplexity table with bars, which shows where a text is unusually predictable
- a methods and limits section

The two building blocks can also be run alone and print JSON:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python binoculars.py draft.txt
HF_HUB_OFFLINE=1 .venv/bin/python perplexity.py Qwen/Qwen2.5-1.5B draft.txt
```

## How it works

**Binoculars** scores a text as the ratio of its perplexity under a "performer" model to the cross-perplexity between an
"observer" model and the performer. The observer-performer cross-entropy captures how surprising the text is *to a language
model that is reading another language model's predictions*, which normalises away prompts and topics that make text
predictable for reasons other than being machine-written. Texts that a model finds much more predictable than this baseline
score low. This repo uses the paper's reference pairing, `tiiuae/falcon-7b` as observer and `tiiuae/falcon-7b-instruct` as
performer, and the paper's thresholds. `binoculars.py` is about 40 lines; the formula follows Section 3 of the paper.

**Perplexity and burstiness** measure how predictable a text is to a language model, and how much that predictability varies
from sentence to sentence. Machine-generated text tends to be uniformly predictable; human writing swings. The idea goes back
to GLTR (Gehrmann et al., 2019) and the GPT-2 output-detector work (Solaiman et al., 2019), and is the basis of commercial
tools such as GPTZero. This repo computes document perplexity, mean sentence perplexity and the standard deviation across
sentences under two models, `Qwen/Qwen2.5-1.5B` and `openai-community/gpt2`.

## Limits

No detector proves authorship. Edited or paraphrased machine text can pass. Short texts, formulaic genres, lists, and text
dense with names and numbers can score as machine-like when written by a person. Both methods are weaker on text that the
underlying models have memorised (famous passages) and on languages other than English. Use the scores alongside drafts,
sources and the writer's history, never as a verdict on their own.

The Binoculars thresholds are specific to the Falcon-7B pair. If you swap models, re-derive thresholds on labelled data.

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
- Ebtesam Almazrouei et al. **The Falcon Series of Open Language Models.** 2023.
  [arXiv:2311.16867](https://arxiv.org/abs/2311.16867). Models: `tiiuae/falcon-7b`, `tiiuae/falcon-7b-instruct` (Apache 2.0).
- Qwen Team. **Qwen2.5 Technical Report.** 2024. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).
  Model: `Qwen/Qwen2.5-1.5B` (Apache 2.0).
- Alec Radford et al. **Language Models are Unsupervised Multitask Learners.** 2019. Model: `openai-community/gpt2` (MIT).

## Controls

`controls/human-letter-1913.txt` is the opening of a letter by Elinore Pruitt Stewart from *Letters of a Woman Homesteader*
(1914), Project Gutenberg #16623, public domain. `controls/ai-marketing-paragraph.txt` is a generic marketing paragraph
generated by Claude Fable 5.1 (Anthropic) in October 2026, unedited. See [`controls/README.md`](controls/README.md).

## License

MIT. Model weights are downloaded separately under their own licenses.
