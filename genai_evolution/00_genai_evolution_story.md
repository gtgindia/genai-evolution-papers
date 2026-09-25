# The Evolution of GenAI: How 15 Papers Taught Machines to Read, Talk, See, Behave, and Reason

This is a teaching guide that connects all 15 landmark papers into one story. Each paper has its own detailed note in this folder (files `01` through `15`). This file is the big picture: what each paper really did, why the next paper was needed, and how they fit together.

Think of the story in four acts. Act 1 builds the language engine. Act 2 fixes bigness and adds eyes. Act 3 teaches manners and step-by-step thinking. Act 4 makes everything cheap, open, and deeply reasoning.

---

## 0. Timeline at a glance

| # | Paper | Year | Core in one line | What changed |
|---|-------|------|------------------|--------------|
| 01 | Attention Is All You Need | 2017 | Replace recurrence with multi-head self-attention | The Transformer backbone for everything after |
| 02 | BERT | 2018 | Bidirectional masked pretraining + fine-tune | Machines learn to understand by reading both ways |
| 03 | GPT-3 | 2020 | 175B decoder does few-shot via prompting | From train-for-task to prompt-for-task |
| 04 | RAG | 2020 | Attach Wikipedia index to generator | Do not memorize everything, look it up |
| 05 | DDPM | 2020 | Learn to reverse 1000-step noising | Stable high-quality image generation without GAN tricks |
| 06 | CLIP | 2021 | Contrast 400M image-text pairs | Words become a universal classifier for pictures |
| 07 | Latent Diffusion | 2021-22 | Diffuse in compressed latent + CLIP conditioning | Stable Diffusion: fast, cheap, open image gen |
| 11 | LoRA | 2021 | Freeze weights, learn tiny low-rank update | Fine-tune giants on one GPU |
| 10 | Chinchilla | 2022 | Scale tokens and params equally | 70B well-fed beats 280B starved; data matters as much as size |
| 12 | FlashAttention | 2022 | Exact attention with tiling in SRAM | 2-4x faster, linear memory, long context |
| 08 | InstructGPT | 2022 | SFT + reward model + PPO | Alignment beats scale; direct parent of ChatGPT |
| 09 | Chain-of-Thought | 2022 | Prompt with 8 worked reasonings | Thinking aloud emerges past ~100B |
| 13 | LLaMA 2 | 2023 | Open 7-70B on 2T tokens + full RLHF + safety | Near-ChatGPT quality with downloadable weights |
| 14 | DPO | 2023 | Same RLHF goal as one classification loss | Alignment without reward model or PPO |
| 15 | DeepSeek-R1 | 2025 | Pure RL with rule rewards + GRPO + distillation | Reasoning is incentivized, then shared with small models |

Three threads run through all of them: cost goes from brute force to clever efficiency, memory goes from closed memorization to open retrieval, and access goes from closed APIs to open weights.

---

## Act 1: How machines learned language (01, 02, 03)

### 01 Attention Is All You Need — the new engine

Before 2017, sequence models read word by word like a person reading with a finger, using recurrent networks or convolutions. That was slow and forgetful over long sentences. The Transformer threw away recurrence and let every word directly look at every other word with scaled dot-product attention, done in parallel heads with positional signals and residual connections.

A tiny 100M model trained in 12 hours on 8 GPUs beat the best translation systems. The trivia is lovely: author order is random with an equal-contribution footnote, the title riffs on the Beatles song All You Need Is Love, Transformer was named because an author liked toy robots, and all eight authors later left Google to found startups like Cohere, Character.AI, and Sakana AI.

Everything after is a Transformer variant. See `01_attention_is_all_you_need.md` for the full gentle derivation.

### 02 BERT — learning to understand by reading both ways

Given the Transformer, there were two roads. BERT took the encoder road for understanding. It hid 15 percent of words and asked the model to fill them using left and right context, plus whether sentence B follows sentence A, pretrained on 3.3B words, then fine-tuned with just a new output head.

One model set records on GLUE, SQuAD, and SWAG with about an hour of fine-tuning. Google even put it into Search. The naming joke continued Sesame Street Muppetware started by ELMo: BERT is Bert. See `02_bert.md`.

BERT is brilliant at understanding but cannot easily generate stories, because it is not autoregressive.

### 03 GPT-3 — learning to generate by predicting next word at scale

GPT-3 took the decoder road for generation. Same Transformer skeleton, but 175B parameters, 96 layers, 300B tokens, and no fine-tuning: you simply describe the task with zero, one, or many examples in the prompt and the model continues.

Few-shot TriviaQA beat fine-tuned records, and humans could not tell its news from human news. At 10x anything before it, trained on Microsoft V100s, it created prompt engineering as a job. But it hallucinated, went stale, followed instructions poorly, needed huge money, and was served as a closed API. See `03_gpt3.md`.

So Act 1 ends with a fork that matters for the whole story: `01 -> 02` for understanding versus `01 -> 03` for generation. Generation won the product race, but its flaws call the next acts.

---

## Act 2: Fixing bigness and adding eyes (10, 04, 05, 06, 07)

### 10 Chinchilla — we scaled wrong

OpenAI scaling laws said 10x compute means 5.5x params but only 1.8x data, so labs built ever-bigger underfed giants like GPT-3 175B on 300B tokens and Gopher 280B on 300B tokens. DeepMind trained 400+ models and found both should scale equally, about 20 tokens per parameter.

Chinchilla 70B on 1.4T tokens beat Gopher 280B, GPT-3 175B, and MT-NLG 530B on MMLU 67.6 percent and much more, while being 4x cheaper to serve. The name is a joke: Gopher versus Chinchilla, both rodents. Lesson: smart data beats dumb size. LLaMA 2 later takes this to heart with 2T tokens. See `10_chinchilla.md`.

Cross-link: `03 -> 10 -> 13`. GPT-3 proved scale, Chinchilla corrected scale, LLaMA 2 applied it openly.

### 04 RAG — do not memorize what you can look up

GPT-3 as closed-book memorizer must store the world in weights, so it invents facts and cannot update without retraining. RAG gives BART a Wikipedia index via frozen DPR retriever plus FAISS search and marginalizes over top retrieved docs, with RAG-Sequence using one doc per answer and RAG-Token mixing per token.

It beat 11B closed-book T5 on Natural Questions and TriviaQA, generated more specific Jeopardy questions preferred by humans, and allowed hot-swapping the index to update knowledge. Today RAG means any retriever plus LLM. See `04_rag.md`.

Cross-link: `03 -> 04`. The brilliant talker gets a librarian and fact-checker. This parametric versus non-parametric tradeoff never goes away and returns in every assistant with browsing.

### 05 DDPM, 06 CLIP, 07 Latent Diffusion — learning to see and dream

While language scaled, vision took a parallel path that later merges.

DDPM revived a ignored 2015 thermodynamics idea: destroy an image with 1000 steps of Gaussian noise, then learn a U-Net to predict the added noise with a simple regression loss. Stable training beat GANs on CIFAR-10 FID 3.17, but sampling needed 1000 sequential steps. Three Berkeley authors sparked the image revolution. See `05_ddpm.md`.

CLIP connected words to pictures. OpenAI collected 400M web image-text pairs and trained image and text encoders contrastively: in a 32k batch, push N correct diagonal pairs together and N-squared-minus-N wrong pairs apart. At test time, embed a photo of a cat for each label and pick the closest. Zero-shot it matched ResNet-50 on ImageNet with no labels and understood memes, OCR, and geography, while showing bias. Backbone for DALL-E and Stable Diffusion text encoder. See `06_clip.md`.

Latent Diffusion fused them. Pixel diffusion cost hundreds of GPU-days, so CompVis in Munich trained an autoencoder first and ran DDPM denoising in small latent space with cross-attention for CLIP text. Same quality at a fraction of cost, runnable on 10GB GPUs. With Stability AI compute and LAION-5B, this became Stable Diffusion v1 open weights in August 2022, the first open rival to DALL-E 2. See `07_latent_diffusion.md`.

Cross-link: `05 + 06 -> 07`. Denoiser plus word-picture dictionary in compressed space equals modern image generation. Teaching point: language ideas unlock vision, and efficiency unlocks openness.

---

## Act 3: Teaching manners and thinking (08, 09, 14)

### 08 InstructGPT — alignment beats scale

GPT-3 could write but missed intent, made up closed-domain facts 41 percent of the time, and obeyed poorly. OpenAI hired about 40 contractors for three steps: 13k demonstrations for supervised fine-tuning, 33k ranked prompts to train a 6B reward model that scores answers, then PPO with KL penalty plus pretraining mix for 256k episodes.

A tiny 1.3B InstructGPT was preferred over 175B GPT-3, 175B InstructGPT won 85 percent versus GPT-3, hallucinations halved to 21 percent, TruthfulQA doubled. Cost was under 2 percent of pretraining. Quietly released in January 2022, it became ChatGPT in November 2022 almost unchanged. See `08_instructgpt.md`.

Cross-link: `03 -> 08`. Capability without obedience becomes a product with obedience. This is the parent of the whole alignment chain.

### 09 Chain-of-Thought — show your work

Even aligned models flopped on grade-school math with standard prompting. Nine Google researchers simply prompted frozen LaMDA, GPT-3, and PaLM 540B with 8 question-plus-rationale-plus-answer examples, so the model writes scratch-paper before answering.

PaLM jumped on GSM8K from 17.9 to 56.9 percent, sports understanding from 80.5 to 95.4 beating human fans, last-letter 99.4 percent. But only past about 100B; small models got worse with fluent nonsense. Sundar Pichai demoed it at I/O 2022. It spawned self-consistency, tree-of-thought, and ReAct. See `09_chain_of_thought.md`.

Cross-link: `03 -> 09`, tested on `08` variants. Prompting unlocks latent reasoning without retraining, but only if the model is big enough and instruction-following.

### 14 DPO — alignment without the pain

RLHF worked but needed four models, online sampling, and fragile PPO tuning. DPO proves the KL-constrained RL optimum has closed form, so reward equals beta times log-ratio of policy over reference plus a constant that cancels in Bradley-Terry choice. You get a simple binary loss on winner versus loser that pushes up good answers and pushes down bad ones, weighted by how wrong you currently are.

On IMDb it dominates PPO on reward versus KL, on TL;DR it wins 61 versus 57 percent for PPO with flat temperature robustness, on dialogue it is the only offline method beating human-chosen answers, and it generalizes better to CNN news. Ten lines of code, beta 0.1, almost no tuning. Zephyr 7B with distilled DPO then beat LLaMA-2-Chat 70B, and Tulu 2 lifted 70B from 86.6 to 95.1 percent on AlpacaEval. Limits are real: offline shift, overfitting, verbosity about 2x longer, noisy labels, needing a good reference. See `14_dpo.md`.

Cross-link: `08 -> 14`. Same human data, same optimum, far simpler. This is what let small open labs align like giants.

---

## Act 4: Cheap, open, reasoning for all (11, 12, 13, 15)

### 11 LoRA and 12 FlashAttention — efficiency is access

Two efficiency papers underlie the open era.

LoRA freezes the giant weight and learns delta as B times A with rank 1 to 8. For GPT-3 175B that is 10,000x fewer trainables, 1.2TB to 350GB VRAM, 350GB to 35MB checkpoint, zero merged latency. It matched full fine-tuning on GLUE and enabled a library of swappable adapters in HuggingFace PEFT, including QLoRA on consumer GPUs. See `11_lora.md`.

FlashAttention is IO-aware, not FLOP-aware. Attention is memory-bound between slow HBM and fast SRAM, so tiling plus online softmax plus recomputation in a fused kernel, never materializing NxN, gives exact attention with linear memory. Up to 7.6x attention speedup, BERT 15 percent faster than MLPerf record, GPT-2 3x faster, first Transformer to solve Path-X 16k. PhD student Tri Dao became famous; v2 and v3 are now defaults. See `12_flashattention.md`.

Together they mean: train cheaply and serve long contexts cheaply. Without them, 13 to 15 stay lab demos.

### 13 LLaMA 2 — open weights done right

Before LLaMA 2, powerful assistants were closed while open bases were raw and unsafe. Meta released 7 to 70B on 2T tokens with 4k context and grouped-query attention, then SFT on 27.5k quality demos plus dual helpful and safety reward models on 2.9M comparisons with iterative rejection sampling plus PPO, Ghost Attention for system memory, and deep red-teaming.

70B base hit MMLU 68.9 near GPT-3.5, chat won 36 percent and tied 31.5 percent versus ChatGPT, ToxiGen fell to near zero. Backstory matters: LLaMA 1 leaked via 4chan torrent and spawned Alpaca and Vicuna on MacBooks, proving demand; LLaMA 2 answered with commercial license via Microsoft and HuggingFace, 150k requests in a week. Debate over truly open source versus open weights started here. See `13_llama2.md`.

Cross-link: `10 + 08/14 + 11/12 -> 13`. Chinchilla tokens plus InstructGPT recipe simplified by DPO thinking plus LoRA and memory efficiency, all published openly.

### 15 DeepSeek-R1 — reasoning is incentivized, not just imitated

The finale fuses everything. Starting from V3-Base 671B MoE with 37B active on 14.8T tokens, R1-Zero uses pure RL with no supervised chains: only think tags plus rule rewards for boxed math, passing code, and format, optimized with GRPO that compares groups of 16 without a value net. Reflection emerges untaught with wait, let me recheck moments, length grows to 20k tokens, AIME 15.6 to 77.9 percent.

R1 then polishes with cold-start SFT plus reasoning RL plus 800k rejection-sampled SFT plus all-scenario RL, reaching AIME 79.8 tied with o1, MATH 97.3 above o1, Codeforces 96.3 percentile tied with o1, ArenaHard 92.3. Distillation then teaches six small Qwen and Llama students with pure SFT; 32B hits AIME 72.6 versus 47.0 for direct RL, even 1.5B beats GPT-4o on math. MIT-licensed, 20-50x cheaper API, top of App Store, Nvidia minus 589B dollars in a day on efficiency fears. Limits remain: no tools, overthinking, language mixing, prompt sensitivity. See `15_deepseek_r1.md`.

Cross-link: `09 -> 15` turns prompted scratch-paper into learned 20k-token verification, `08/14 -> 15` turns human preference PPO into rule GRPO, `10/12 -> 15` affords it.

---

## How the threads tie together

**Capability versus behavior.** Pretraining from 01 to 03 gives raw capability. Alignment from 08 and 14 gives behavior. Reasoning from 09 and 15 gives verification. Students should never confuse fluent continuation with helpfulness or correctness; each needed its own paper and data.

**Memory: memorize versus look up.** 03 memorizes, 04 retrieves. RAG reduces hallucination and eases updates but adds retrieval engineering. Modern assistants use both: weights for skills, index for facts, reasoning for combining them.

**Scale: bigger versus smarter.** 03 says bigger brings emergence, 10 says underfed bigger wastes money, 13 and 15 say well-fed plus aligned plus distilled wins. Always ask tokens per parameter, not just parameters.

**Efficiency decides who builds.** Latent space in 07, adapters in 11, IO-aware kernels in 12, MoE sparsity plus GRPO in 15 are not footnotes. Each cut cost by 4x to 50x and moved power from three labs to thousands.

**Openness accelerates.** Transformer open, GPT-3 closed API, Stable Diffusion open beats closed image labs, LLaMA leak forces LLaMA 2 open, R1 MIT shocks the market. Open weights plus recipes plus adapters is why you can run this history on a laptop.

**Vision and language converge.** DDPM patience plus CLIP grounding plus latent thrift is the same pattern as language: stable objective plus broad supervision plus efficiency. Future agents will retrieve like 04, see like 06 and 07, talk like 03 and 13, behave like 08 and 14, and reason like 09 and 15.

---

## Six lessons to carry forward

1. Architecture enables scale. Attention removed the sequential bottleneck; everything else scaled it. When stuck, fix the bottleneck first like FlashAttention did for memory.
2. Scale brings surprises, then bills. Emergence is real in few-shot and chain-of-thought, but Chinchilla shows data quality and quantity beat raw size. Count tokens, not just params.
3. Do not memorize what you can look up. Use parametric memory for fluency and skills, non-parametric retrieval for fresh facts and provenance.
4. Behavior is a separate layer. InstructGPT and DPO prove the same base can be wild or helpful depending on preference data and loss. Budget for alignment, not just pretraining.
5. Thinking can be elicited then automated. Prompted chains today become trained reasoning tomorrow. Distillation then shares it with small models that could never explore alone.
6. Simplicity and openness win. DPO replaced PPO with ten lines, LoRA replaced full fine-tuning with megabytes, latent diffusion replaced pixel diffusion with consumer GPUs, R1 replaced secret chains with MIT weights. The best idea is the one others can actually run.

Start with `01` through `03` for foundations, dip into `05-07` for vision, study `08-09` then `14-15` for alignment and reasoning, and keep `10-12` beside you whenever cost or memory hurts. That path is the involution of GenAI from predicting words to reliably helping with the world.
