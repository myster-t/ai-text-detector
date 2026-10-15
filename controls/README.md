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
| `darwin-origin-of-species-1859.txt` | Charles Darwin, *On the Origin of Species* (1859), ch. 4. Project Gutenberg #1228 | 441 | public domain |
| `thoreau-walden-1854.txt` | Henry David Thoreau, *Walden* (1854). Gutenberg #205 | 443 | public domain |
| `douglass-narrative-1845.txt` | Frederick Douglass, *Narrative of the Life of Frederick Douglass* (1845). Gutenberg #23 | 445 | public domain |
| `twain-life-on-the-mississippi-1883.txt` | Mark Twain, *Life on the Mississippi* (1883), ch. 4. Gutenberg #245 | 442 | public domain |
| `austen-persuasion-1817.txt` | Jane Austen, *Persuasion* (1817), ch. 1. Gutenberg #105 | 308 | public domain |
| `emerson-essays-1841.txt` | Ralph Waldo Emerson, "Self-Reliance", *Essays, First Series* (1841). Gutenberg #2944 | 353 | public domain |
| `dubois-souls-of-black-folk-1903.txt` | W. E. B. Du Bois, *The Souls of Black Folk* (1903), ch. 2. Gutenberg #408 | 300 | public domain |
| `franklin-autobiography-1791.txt` | Benjamin Franklin, *Autobiography* (written 1771–1790). Gutenberg #20203 | 398 | public domain |
| `keynes-economic-consequences-1919.txt` | John Maynard Keynes, *The Economic Consequences of the Peace* (1919). Gutenberg #15776 | 410 | public domain |
| `chopin-the-awakening-1899.txt` | Kate Chopin, *The Awakening* (1899). Gutenberg #160 | 339 | public domain |
| `shelley-frankenstein-1818.txt` | Mary Shelley, *Frankenstein* (1818), ch. 4. Gutenberg #84 | 443 | public domain |
| `washington-up-from-slavery-1901.txt` | Booker T. Washington, *Up from Slavery* (1901). Gutenberg #2376 | 416 | public domain |
| `wells-war-of-the-worlds-1898.txt` | H. G. Wells, *The War of the Worlds* (1898). Gutenberg #36 | 426 | public domain |
| `faraday-chemical-history-of-a-candle-1861.txt` | Michael Faraday, *The Chemical History of a Candle* (1861), lecture I. Gutenberg #14474 | 399 | public domain |
| `grimm-fairy-tales-1812.txt` | Grimms' *Fairy Tales*, English translation (1884 ed.). Gutenberg #2591 | 379 | public domain |
| `carroll-alice-in-wonderland-1865.txt` | Lewis Carroll, *Alice's Adventures in Wonderland* (1865), ch. 1. Gutenberg #19033 | 343 | public domain |
| `dostoevsky-brothers-karamazov-1880.txt` | Fyodor Dostoevsky, *The Brothers Karamazov*, Constance Garnett translation (1912). Gutenberg #28054 | 434 | public domain |
| `kant-critique-of-pure-reason-1781.txt` | Immanuel Kant, *Critique of Pure Reason*, Meiklejohn translation (1855). Gutenberg #4280 | 306 | public domain |
| `machiavelli-the-prince-1532.txt` | Niccolò Machiavelli, *The Prince*, W. K. Marriott translation (1908), ch. 3. Gutenberg #1232 | 425 | public domain |
| `hc3-human-open-qa-7.txt`, `-8.txt` | Open-domain Q&A answers (Wikipedia-derived, pre-2022), from HC3 | 186–192 | CC BY-SA 4.0 |
| `hc3-human-reddit-eli5-9.txt` | Reddit r/explainlikeimfive answer (pre-2022), from HC3 | 224 | CC BY-SA 4.0 |
| `hc3-human-finance-10.txt` | Personal-finance Stack Exchange answer, from HC3 | 212 | CC BY-SA 4.0 |
| `hc3-human-medicine-11.txt` | Doctor's answer on a medical Q&A site, from HC3 | 220 | CC BY-SA 4.0 |
| `raid-arxiv-abstract-1.txt`, `-2.txt` | Human-written arXiv abstracts (pre-2023), the human rows of the RAID benchmark | 200–400 | MIT (RAID) |

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
| `bench2026-claude-3.5-sonnet.txt` | Claude 3.5 Sonnet (Oct 2024), research-style history article, from the 2026 AI Detector Benchmark | 427 | MIT |
| `bench2026-claude-3.7-sonnet.txt` | Claude 3.7 Sonnet, nutrition article, same source | 381 | MIT |
| `bench2026-claude-sonnet-4.txt` | Claude Sonnet 4, essay on supply chains, same source | 443 | MIT |
| `bench2026-claude-sonnet-4.5.txt` | Claude Sonnet 4.5, finance blog post, same source | 418 | MIT |
| `bench2026-deepseek-v3.txt` | DeepSeek V3 (deepseek-chat), motivational speech, same source | 385 | MIT |
| `bench2026-gpt-3.5-turbo.txt` | GPT-3.5 Turbo (0613), legal analysis, same source | 394 | MIT |
| `bench2026-gpt-4.1.txt` | GPT-4.1, essay on supply chains, same source | 414 | MIT |
| `bench2026-gpt-4o.txt` | GPT-4o, science-fiction story, same source | 439 | MIT |
| `bench2026-gpt-5-chat.txt` | GPT-5 (chat), speech on entrepreneurship, same source | 442 | MIT |
| `bench2026-gpt-5-mini.txt` | GPT-5 mini, argumentative essay on AI and labour, same source | 427 | MIT |
| `bench2026-grok-4.txt` | Grok 4, essay on supply chains, same source | 431 | MIT |
| `bench2026-kimi-k2.txt` | Kimi K2 (0905 preview), article on circadian rhythm, same source | 306 | MIT |
| `bench2026-o3.txt` | OpenAI o3, essay on global warming, same source | 414 | MIT |
| `qwen3.5-9b-product-review.txt` | Qwen3.5-9B (instruct, run locally in this repo, October 2026), customer review, unedited | 340 | MIT (this repo) |
| `qwen3.5-9b-history-explainer.txt` | Qwen3.5-9B, local, explainer on the Roman Republic, unedited | 371 | MIT |
| `qwen3.5-9b-travel-blog.txt` | Qwen3.5-9B, local, first-person travel post, unedited | 389 | MIT |
| `raid-mistral-7b-instruct-abstract.txt` | Mistral-7B-Instruct (2023), scientific abstract from a title, RAID benchmark | 330 | MIT (RAID) |
| `raid-mpt-30b-chat-abstract.txt` | MPT-30B-Chat (2023), scientific abstract, RAID | 331 | MIT (RAID) |
| `raid-llama-2-70b-chat-abstract.txt` | Llama-2-70B-Chat (2023), scientific abstract, RAID | 330 | MIT (RAID) |
| `raid-gpt-2-xl-abstract.txt` | GPT-2 XL (2019), scientific abstract, RAID | 330 | MIT (RAID) |

2026 AI Detector Benchmark: [mattc95/2026-AI-DETECTOR-BENCHMARK](https://github.com/mattc95/2026-AI-DETECTOR-BENCHMARK)
(MIT), 500 texts generated in February 2026 from 13 models with themed prompts; one plain-prose text per model was taken,
markdown emphasis and heading marks removed, nothing else edited. Public-domain passages were cut from the Project Gutenberg
plain-text editions at a paragraph boundary a few percent into each book and trimmed to at most 450 words at a sentence end.

RAID: Dugan et al., *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors* (ACL 2024),
[arXiv:2405.07940](https://arxiv.org/abs/2405.07940), dataset `liamdugan/raid` on Hugging Face, MIT. Rows with no
adversarial attack were taken from the abstracts domain (the only domain in the first shard); the Qwen3.5 texts were
generated by `Qwen/Qwen3.5-9B` with sampling (temperature 0.7, top-p 0.9) from one-line prompts asking for plain prose.

HC3: Guo et al., *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection* (2023),
[arXiv:2301.07597](https://arxiv.org/abs/2301.07597), dataset `Hello-SimpleAI/HC3` on Hugging Face, CC BY-SA 4.0. Reddit
spacing artefacts in the human answers (" ," " 's") were normalised; nothing else was edited.

The set now holds 35 human and 32 machine texts. Known weaknesses: the public-domain passages may be partly familiar to the
language models (lowering their perplexity); the 2023 open-model texts are all scientific abstracts; the Qwen3.5 texts are
scored by Qwen3.5 models under the default profile, the easiest case for a probability detector. Add your own files to either folder; they are picked up
automatically.
