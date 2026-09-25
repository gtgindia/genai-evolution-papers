# LoRA: Low-Rank Adaptation of Large Language Models — Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen, Microsoft, 2021 — arXiv:2106.09685

## The story in one paragraph

This paper solves the painful problem of customizing a giant language model without copying the whole giant every time. The authors freeze all of the original pre-trained weights and learn only a tiny extra change that is built from two very small matrices multiplied together. Because those two matrices have a very small inner size called the rank, they contain only a few million numbers instead of hundreds of billions. The paper shows that this trick, called Low-Rank Adaptation or LoRA, matches or beats full fine-tuning on RoBERTa, DeBERTa, GPT-2, and even GPT-3 with one hundred and seventy-five billion parameters, while cutting trainable parameters by ten thousand times, cutting memory by about three times, and adding zero extra delay when answering questions.

## What problem was hurting before this paper

Before LoRA, adapting a model usually meant full fine-tuning, which means updating every single parameter for every new task. For a small model like RoBERTa this was merely annoying, but for GPT-3 with one hundred and seventy-five billion parameters it became impossible in practice. Each fine-tuned copy needed about three hundred and fifty gigabytes just to store, and about one point two terabytes of video memory to train with the Adam optimizer. If a company wanted one hundred different customers or tasks, it would need about thirty-five terabytes of storage and slow switching between models.

Researchers had tried cheaper tricks, but each had a catch. Adapter layers added extra neural-network layers between the old layers, which forced the computer to do extra sequential steps and added up to thirty percent extra delay when answering one question at a time, which is exactly how chat services work. Prefix-tuning and prompt-tuning reserved part of the input sentence for special trainable words, which stole space from the real task and was hard to optimize, with performance that jumped up and down as you added more special words. The field needed a method that was tiny to store, fast to train, and completely free at inference time.

## The core idea, explained simply (with an analogy)

The core idea is that the change a model needs for a new task lives in a very small subspace, even though the model itself is huge. Instead of learning a giant change matrix, LoRA learns two skinny matrices whose product equals that change.

A helpful analogy is redecorating a huge office building. Full fine-tuning is like rebuilding every wall for every new tenant. Adapters are like adding new hallways that everyone must walk through, which slows people down. LoRA is like leaving the whole building exactly as it is and giving each tenant a tiny set of sticky notes that say how to adjust each room slightly. When the tenant moves in, you simply add the sticky-note instructions to the walls in one quick step, so walking through the building afterward is just as fast as before. When a new tenant arrives, you peel off the old notes and stick on new ones, which takes almost no time or storage.

In math words, if the frozen weight is W-zero with shape d by k, LoRA learns B with shape d by r and A with shape r by k, where r is tiny like one, two, four, or eight, while d can be as large as twelve thousand two hundred and eighty-eight. The update is simply B times A, and the final weight used for answering is W-zero plus B times A.

## How it actually works (step by step: method, math explained in words, experiments — explain each piece in words; explain key equations in plain language)

The paper starts from normal language-model training. You have a pre-trained model with weights called Phi-zero, and a downstream dataset of pairs like questions with answers or articles with summaries. Full fine-tuning would move Phi-zero to Phi-zero plus Delta-Phi by following gradients to make the correct next words more likely. The problem is that Delta-Phi is as big as Phi-zero itself.

LoRA rewrites Delta-Phi as a function of a tiny parameter set called Theta. The training goal is still to make the correct next words more likely, but now you only optimize Theta, while Phi-zero never moves. In plain words, you freeze the giant and only train the sticky notes.

The forward pass is beautifully simple. Normally a layer computes h equals W-zero times x, where x is the incoming signal and h is the outgoing signal. With LoRA it computes h equals W-zero times x plus B times A times x. At the start of training, A is filled with small random Gaussian noise and B is filled with zeros, so B times A starts as exactly zero and the model behaves exactly like the original. During training, only A and B receive gradient updates. The term B times A times x is scaled by alpha divided by r, where alpha is a constant. In words, this scaling keeps learning stable when you change r, so you do not need to retune the learning rate every time. In practice the authors set alpha equal to the first r they try and leave it alone.

Because the design is just linear addition, you can merge the result for deployment. You compute W equals W-zero plus B times A once, store that sum, and run the model exactly as if it were fully fine-tuned, with no extra layers and no extra delay. To switch tasks, you subtract the old B times A and add the new one, which is a tiny and fast operation.

The authors chose to apply LoRA mainly to the attention matrices called W-q, W-k, W-v, and W-o, which handle queries, keys, values, and output projections. They froze the feed-forward MLP blocks for simplicity and efficiency. Through careful tests on GPT-3 with a budget of eighteen million trainable parameters, they found that adapting both W-q and W-v together works best. Putting all of the budget into only W-q or only W-k performed clearly worse. This suggests that it is better to lightly touch two kinds of attention weights than to heavily touch only one.

The key equation for the loss says to maximize the sum over all training pairs and over all words in each answer of the log probability of each correct word given the question and the previous correct words, where the probability comes from the frozen model plus the LoRA update. In plain words, you reward the model whenever its tiny sticky notes help it guess the next correct word.

The paper also studies the subspace overlap in detail. They take the A matrix learned with rank eight and the A matrix learned with rank sixty-four, break each into its singular directions, which are like the most important patterns inside, and measure how much those patterns overlap with a Grassmann distance score between zero and one. They find that the very top direction overlaps strongly with similarity above zero point five, while other directions look like random noise. They also compare two different random seeds and find the same top direction appears both times. This is strong evidence that the useful change really is low-rank.

They also compare the size of the update to the original weight. They project W onto the subspace of Delta-W and measure Frobenius norms, which are like total energies. For rank four they find that the projection energy from W is only zero point three two, while the energy of Delta-W itself is six point nine one, which is an amplification of about twenty-one times. In words, LoRA does not repeat the loudest patterns already in W. It strongly boosts quiet patterns that were present but ignored during pre-training, because those quiet patterns matter for the new task.

The experiments cover four model families. On the GLUE language-understanding benchmark with RoBERTa-base at one hundred and twenty-five million parameters, LoRA with only zero point three million trainable parameters reached an average of eighty-seven point two, beating full fine-tuning at eighty-six point four and all adapter variants. On RoBERTa-large at three hundred and fifty-five million parameters, LoRA with zero point eight million parameters reached eighty-nine point zero, again beating full fine-tuning at eighty-eight point nine. On DeBERTa-XXL at one point five billion parameters, LoRA with four point seven million parameters reached ninety-one point three, beating full fine-tuning at ninety-one point one.

On GPT-2 medium and large for the E2E restaurant-description challenge, LoRA with zero point three five million parameters reached a BLEU score of seventy point four, beating full fine-tuning at sixty-eight point two and both adapter and prefix-layer baselines. BLEU, NIST, METEOR, ROUGE-L, and CIDEr are all automatic scores where higher means the generated description matches human references more closely.

On GPT-3 with one hundred and seventy-five billion parameters, LoRA with only four point seven million parameters reached seventy-three point four percent on WikiSQL, ninety-one point seven percent on MNLI-matched, and ROUGE scores of fifty-three point eight, twenty-nine point eight, and forty-five point nine on the SAMSum dialogue-summarization set. With thirty-seven point seven million parameters it reached seventy-four point zero percent on WikiSQL. In all cases it matched or beat full fine-tuning, BitFit, prefix-embedding, prefix-layer, and adapter baselines, despite using up to ten thousand times fewer trainable numbers.

## Key results and numbers from the paper (with the actual scores/tables described in words)

The most striking numbers are about efficiency. Compared to GPT-3 fine-tuned with Adam, LoRA cuts trainable parameters by ten thousand times and cuts video-memory use during training from one point two terabytes to three hundred and fifty gigabytes, which is about a three-times saving. The checkpoint for one task shrinks from three hundred and fifty gigabytes to about thirty-five megabytes when using rank four on query and value matrices. That means one hundred customized models need only about three hundred and fifty-four gigabytes instead of thirty-five terabytes. Training speed on GPT-3 improves by about twenty-five percent, from thirty-two point five tokens per second per V100 graphics card to forty-three point one tokens per second, because gradients are not computed for the frozen giant.

On quality, the GLUE table shows RoBERTa-base with LoRA winning on SST-2 sentiment with ninety-five point one percent, on RTE entailment with eighty-six point six percent versus seventy-eight point seven percent for full fine-tuning, and on STS-B similarity with ninety-one point five. RoBERTa-large with LoRA wins on MNLI inference with ninety point six percent and on QNLI with ninety-four point nine percent.

The inference-latency table shows why adapters hurt. On GPT-2 medium with batch size one and sequence length one hundred and twenty-eight, adapters add twenty point seven to thirty point three percent extra delay, while LoRA adds zero because it is merged.

The rank study shows that rank one already works surprisingly well. On WikiSQL, adapting both query and value with rank one gives seventy-three point four percent, rank four gives seventy-three point seven percent, and rank sixty-four gives seventy-three point five percent. On MNLI, rank one gives ninety-one point three percent and rank eight gives ninety-one point six percent. In plain words, one or two directions capture almost all of the useful change for these tasks.

## Why this paper matters (what later work it unlocked — name the specific later papers among the 15 where relevant)

LoRA became the standard way to customize large models without a supercomputer. The LLaMA-2 paper, which is number thirteen in our set, is routinely fine-tuned with LoRA and its stronger cousin QLoRA by the open-source community, which made chat versions, medical versions, and code versions possible on a single graphics card. The InstructGPT paper, which is number eight, showed that human-feedback fine-tuning is powerful but expensive, and LoRA made that kind of alignment affordable for everyone else. The DPO paper, which is number fourteen, almost always uses LoRA in public implementations because preference optimization needs to train efficiently on small curated sets. The DeepSeek-R1 paper, which is number fifteen, builds on efficient adaptation culture where small rank updates and careful data use let reasoning abilities emerge without retraining the whole giant from scratch. Beyond our fifteen, LoRA directly enabled QLoRA, LoRA-XS, DoRA, and the entire HuggingFace PEFT library, plus thousands of tiny downloadable adapters for diffusion models that started from the Latent Diffusion paper, which is number seven.

## Honest limitations

The authors admit that LoRA is not magic for every situation. If the new task is in a very different language or domain from pre-training, a tiny rank like one or two may not be enough, and you might need full fine-tuning or a much larger rank. They mostly test only attention weights and freeze the MLP blocks, layer norms, and biases, leaving open whether those parts also need adaptation for harder tasks. There is also a batching problem. If you merge B times A into W for speed, you cannot easily serve many different tasks in the same batch, because each example would need a different W. You can keep the adapters separate and choose them per example, but then you lose some speed. Finally, choosing which matrices to adapt and which rank to use still relies on trial and error rather than a proven theory.

## Fun trivia and history (from your web search)

The inventor story is unusually clear because first author Edward Hu recorded a video explainer. He says that in early 2021, just after GPT-3 came out, his team at Microsoft was asked a blunt business question, which was whether this GPT-3 stuff can actually make money. They discovered that few-shot prompting was not reliable enough for real products like natural-language-to-code, which rarely appears in training data, so fine-tuning was necessary. But a single GPT-3 checkpoint was about one terabyte when counting optimizer states, which took minutes to load and cost a fortune to store per customer. They tried existing efficient methods and found each one compromised on quality or speed, so they invented LoRA with product impact in mind.

The paper first appeared on arXiv in June 2021 as version one, then a stronger version two in October 2021 added GLUE results and better baselines, and it was finally published at ICLR 2022. The GitHub project microsoft/LoRA released a tiny library called loralib with examples for GPT-2, RoBERTa, and DeBERTa. In February 2023, HuggingFace added LoRA to its PEFT library, which is when usage exploded. Today the community joke is that LoRA turned a one-terabyte model into a twenty-five-megabyte email attachment, because the four point seven million trainable numbers fit easily on a laptop.

## Key terms explained (glossary of 5-8 terms)

Fine-tuning means continuing to train a pre-trained model on a new task by updating its weights with gradients. Full fine-tuning updates everything, while LoRA updates only a tiny extra piece.

Rank means the inner size r of the two small matrices. Rank one means the update is built from a single pattern, rank eight means eight patterns. Smaller rank means fewer numbers to learn.

Intrinsic dimension is the idea that a huge model really lives on a much smaller landscape. Even though GPT-3 has one hundred and seventy-five billion knobs, the useful changes for a new task can be described with only a few million numbers.

Adapter layers are extra neural-network blocks inserted between old layers. They are small but must be executed step by step, which adds delay during live chat.

Prefix-tuning means learning special virtual words that are placed before the real input. Those virtual words take up space that could have been used for the real question, and they can be tricky to optimize.

Inference latency means the extra waiting time when the model answers. LoRA has zero extra latency after merging because the math becomes identical to a fully fine-tuned model.

Checkpoint means the saved file containing all trained numbers. A full GPT-3 checkpoint is hundreds of gigabytes, while a LoRA checkpoint is tens of megabytes.

## Connections (which earlier paper it builds on, which later paper builds on it)

This paper builds on the Attention Is All You Need paper, which is number one, because LoRA specifically modifies the query, key, value, and output matrices inside Transformer attention. It builds on the BERT paper, which is number two, and the GPT-3 few-shot paper, which is number three, because those papers established the pre-train then fine-tune paradigm and showed that few-shot prompting alone is often not enough, which is proven again in the appendix where fine-tuning beats few-shot by a large margin.

Later, the InstructGPT paper, which is number eight, the LLaMA-2 paper, which is number thirteen, the DPO paper, which is number fourteen, and the DeepSeek-R1 paper, which is number fifteen, all rely on LoRA in practice. Open-source chat models based on LLaMA-2 are almost always aligned with LoRA or QLoRA, DPO tutorials use LoRA by default to save memory, and reasoning-distillation efforts in the spirit of DeepSeek-R1 distribute tiny LoRA adapters instead of whole models.
