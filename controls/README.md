# Controls

Every run re-scores these texts with the same models as the file under test, so each report shows where the file sits
among known-human and known-machine writing. Spread across decades, registers and generators on purpose.

## Human (`controls/human/`)

| File | Source | Words | Licence |
| --- | --- | ---: | --- |
| `stewart-letter-1913.txt` | Elinore Pruitt Stewart, *Letters of a Woman Homesteader* (1914), letter opening. Project Gutenberg #16623 | 347 | public domain |
| `addams-hull-house-1910.txt` | Jane Addams, *Twenty Years at Hull-House* (1910), on the opening of the settlement. Project Gutenberg #1325 | ~330 | public domain |
| `bird-rocky-mountains-1879.txt` | Isabella L. Bird, *A Lady's Life in the Rocky Mountains* (1879), first letter. Project Gutenberg #755 | ~330 | public domain |
| `hc3-human-reddit-eli5-1.txt`, `-2.txt` | Reddit r/explainlikeimfive answers (pre-2022), from the HC3 corpus | 204–299 | CC BY-SA 4.0 |
| `hc3-human-finance-3.txt`, `-4.txt` | Personal-finance Stack Exchange answers, from HC3 | 216–410 | CC BY-SA 4.0 |
| `hc3-human-medicine-5.txt`, `-6.txt` | Doctors' answers on a medical Q&A site, from HC3 | 213–238 | CC BY-SA 4.0 |

## Machine (`controls/ai/`)

| File | Generator | Words | Licence |
| --- | --- | ---: | --- |
| `claude-marketing-paragraph.txt` | Claude Fable 5.1 (Anthropic), October 2026, generic marketing copy, unedited | 214 | MIT (this repo) |
| `claude-personal-essay.txt` | Claude Fable 5.1, a college-application-style personal essay, unedited | 270 | MIT |
| `claude-eli5-answer.txt` | Claude Fable 5.1, an "explain like I'm five" answer, unedited | 264 | MIT |
| `claude-business-email.txt` | Claude Fable 5.1, a workplace email, unedited | 240 | MIT |
| `claude-short-story.txt` | Claude Fable 5.1, short fiction, unedited | 261 | MIT |
| `claude-technical-explainer.txt` | Claude Fable 5.1, a technical explanation, unedited | 296 | MIT |
| `hc3-chatgpt-reddit-eli5-1.txt`, `-2.txt` | ChatGPT (GPT-3.5, January 2023) answers to the same ELI5 questions, from HC3 | 200–205 | CC BY-SA 4.0 |
| `hc3-chatgpt-finance-3.txt`, `-4.txt` | ChatGPT (GPT-3.5, January 2023), finance questions, from HC3 | 217–231 | CC BY-SA 4.0 |
| `hc3-chatgpt-open-qa-5.txt`, `-6.txt` | ChatGPT (GPT-3.5, January 2023), open-domain questions, from HC3 | 180–233 | CC BY-SA 4.0 |

HC3: Guo et al., *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection* (2023),
[arXiv:2301.07597](https://arxiv.org/abs/2301.07597), dataset `Hello-SimpleAI/HC3` on Hugging Face, CC BY-SA 4.0. Reddit
spacing artefacts in the human answers (" ," " 's") were normalised; nothing else was edited.

Known weaknesses of this set: the public-domain texts may be partly familiar to the language models (lowering their perplexity),
and the machine texts come from only two generator families. Add your own files to either folder; they are picked up
automatically.
