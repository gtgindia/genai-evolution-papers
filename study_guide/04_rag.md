# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020, arXiv:2005.11401)

Authors: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. The work comes from Facebook AI Research with collaborators at University College London and New York University. It was first posted to arXiv in May 2020 and later appeared at NeurIPS 2020.

## The story in one paragraph

This paper asks a simple question. What if a language model could look things up while it writes, instead of trying to memorize the whole world inside its weights. The authors build a system that combines two kinds of memory. One is parametric memory, which is a pre-trained BART generator that stores knowledge in its parameters. The other is non-parametric memory, which is a giant searchable index of Wikipedia that stores knowledge as readable text. For every question, the system first retrieves a handful of helpful Wikipedia passages, and then it generates an answer that is grounded in those passages. The paper shows that this open-book approach beats both giant closed-book models and older extractive question-answering systems on several benchmarks, while also producing answers that people judge as more factual and more specific.

## What problem was hurting before this paper

Before this paper, large pre-trained language models were celebrated as if they were knowledge bases in themselves. Work like the T5 closed-book question-answering experiments showed that a very large model could answer many factual questions from memory alone. But this created three painful problems. First, the knowledge was frozen at training time, so the model could not easily learn that a new president had been elected. Second, the model could not point to where an answer came from, so users could not check its sources. Third, the model often hallucinated, which means it produced smooth and confident sentences that were factually wrong.

On the other side, retrieval-based question-answering systems did look at documents, but they were usually extractive. That means they could only copy a short span of text from a document. They could not combine clues from several documents, and they scored zero if the exact answer string was missing. Earlier hybrid models like REALM and ORQA had tried differentiable retrieval, but they were only tested on extractive question answering. What was missing was a general recipe that gave the workhorse sequence-to-sequence generator its own searchable memory.

## The core idea, explained simply (with an analogy)

Think of a closed-book exam versus an open-book exam. A closed-book student must memorize every fact, and if the textbook changes, the student must study all over again. An open-book student only needs to learn how to find the right page quickly and how to explain it clearly. RAG is the open-book student.

In this analogy, BART is the student's writing and reasoning ability. The Wikipedia index is the library. The DPR retriever is the librarian who fetches the most useful books. The student does not memorize every page. Instead, the student learns to ask the librarian a good question, read the returned pages, and then write a final answer that blends what was read with what was already known. If the world changes, you do not need to retrain the student. You simply replace the old books on the shelf with new books, which the paper calls hot-swapping the index.

## How it actually works (step by step: architecture, training objective, inference — explain each piece in words; explain key equations in plain language)

The system has two learned parts that work together as one probabilistic model.

The retriever decides which documents matter. It is based on Dense Passage Retrieval, usually called DPR. There are two BERT-base encoders. One encodes the question into a vector, and the other encodes each Wikipedia passage into a vector. The score for a passage is simply the dot product between the two vectors, which measures how similar their directions are. In plain language, the model computes a compatibility score that is high when the question and the passage seem to talk about the same thing. To find the best passages quickly among 21 million candidates, the system uses Maximum Inner Product Search with a FAISS index. The paper starts from a DPR retriever that was already trained on Natural Questions and TriviaQA, and it builds the index from a December 2018 Wikipedia dump split into disjoint 100-word chunks.

The generator writes the answer. It is BART-large, a pre-trained encoder-decoder Transformer with about 400 million parameters. To use a retrieved passage, the system simply concatenates the input question with that passage and feeds the combined text into BART. So BART sees both what you asked and what the librarian found.

The paper compares two ways of combining retrieval with generation. In RAG-Sequence, the model picks one passage and uses that same passage to generate the entire answer. The final probability of an answer is a weighted average over the top-K passages, where each passage is weighted by how likely the retriever thought it was. In plain language, the model says that there are several plausible sources, and it hedges by mixing their votes. In RAG-Token, the model is allowed to switch sources word by word. For each new token, it mixes the predictions that come from all the retrieved passages. This is useful when an answer needs facts from two different places, such as a Jeopardy question that mentions two different novels by the same author.

Training is end-to-end without telling the retriever which document is correct. The only supervision is pairs of inputs and desired outputs. The training objective is the negative log-likelihood of the correct answer after marginalizing over retrieved documents. That is a fancy way of saying that the model is punished whenever the weighted mixture assigns low probability to the right answer, and both the generator and the question encoder learn from that punishment. To save computation, the document encoder and the FAISS index are kept frozen. Only the question encoder and the BART generator are fine-tuned. This avoids rebuilding the 21-million-vector index after every update.

At test time, the two variants need different decoding. RAG-Token can be decoded with ordinary beam search because its per-token mixture looks like a normal next-token distribution. RAG-Sequence is trickier because the full answer probability does not factor neatly per token. The paper therefore runs beam search separately for each retrieved document and then re-scores the candidate answers by summing across documents. The exact version is called Thorough Decoding, and a faster approximation that skips extra forward passes is called Fast Decoding.

## Key results and numbers from the paper (with the actual scores/tables described in words)

The paper evaluates on open-domain question answering with Exact Match, which counts an answer as correct only if it matches the gold answer exactly.

On Natural Questions, RAG-Sequence reaches 44.5 and RAG-Token reaches 44.1. This beats DPR at 41.5, REALM at 40.4, and the giant closed-book T5 with 11 billion parameters at only 34.5. On TriviaQA, the paper reports two test splits. On the standard open-domain split, RAG-Sequence gets 56.8 and RAG-Token gets 55.2, which is close to DPR at 57.9. On the Wikipedia test split used by the T5 authors, RAG-Sequence reaches 68.0 and RAG-Token reaches 66.1, clearly above the T5 numbers around 50 to 60. On WebQuestions, RAG-Token gets 45.5 and RAG-Sequence gets 45.2, beating DPR at 41.1 and T5 with salient-span masking at 44.7. On CuratedTrec, RAG-Sequence gets 52.2, ahead of DPR at 50.6. The paper emphasizes that even on these extractive benchmarks, free-form generation wins over copying spans. In fact, on Natural Questions, RAG still gets 11.8 percent right even when none of the retrieved documents contains the exact answer, where an extractive system would necessarily score zero.

For abstractive question answering on MS-MARCO in an open-domain setting without gold passages, RAG-Sequence improves over a BART baseline by about 2.6 BLEU points and 2.6 ROUGE-L points. For the newly proposed Jeopardy question generation task, RAG-Token gets the best Q-BLEU score at 22.2, ahead of BART at 19.7. Human judges compared 452 pairs of BART and RAG generations. They judged RAG as more factual in 42.7 percent of cases and BART as more factual in only 7.1 percent. They also judged RAG as more specific by a wide margin. Generation diversity is higher too: the ratio of distinct trigrams is highest for gold text, next for RAG-Sequence, then RAG-Token, and lowest for BART.

For FEVER fact verification, RAG reaches 72.5 percent on the three-way task and 89.5 percent on the two-way task. This is within a few points of complex pipeline systems that use strong retrieval supervision, even though RAG uses no evidence labels. The top retrieved article matches a gold evidence article 71 percent of the time, and a gold article appears in the top ten 90 percent of the time.

Ablations show that learning the retriever matters. Frozen-retriever and BM25 variants are worse on almost every task, especially open-domain question answering. Retrieving more documents at test time helps RAG-Sequence steadily, while RAG-Token peaks around ten documents on Natural Questions.

## Why this paper matters (what later work it unlocked — name the specific later papers among the 15 where relevant)

This paper gave a clean name and recipe to an idea that now runs much of applied AI. The phrase retrieval-augmented generation, now shortened everywhere to RAG, comes from this paper. Every modern assistant that searches documents before answering, cites sources, or lets you hot-swap a company knowledge base is a descendant of this design.

Among the other landmark papers in this collection, the connection is easy to see. The InstructGPT paper (08) and the LLaMA 2 paper (13) describe powerful parametric chat models that are still frozen in time and prone to hallucination. RAG is the complementary technique that grounds those chat models in fresh external text. The DeepSeek-R1 paper (15) pushes reasoning, but reasoning still needs correct facts to reason about, which is exactly what a retriever supplies. Even the attention, BERT, and GPT-3 papers (01, 02, 03) in this set describe stronger readers and writers, while RAG describes how any such reader-writer can be given a library card. Later industry systems like Atlas, RETRO, REALM follow-ups, LangChain, and LlamaIndex all cite this paper as the starting point for practical open-book language models.

## Honest limitations

The system still depends on Wikipedia as its world. If the answer is not in Wikipedia, or if Wikipedia is biased or out of date, RAG can still fail. The paper notes that some MS-MARCO questions cannot be answered from Wikipedia alone, so the model must fall back on parametric guessing.

Retrieval is not perfect either. The document encoder is frozen, so the index never improves during fine-tuning. The paper also reports a failure mode called retrieval collapse on tasks like story generation, where the retriever learns to return the same documents no matter the input, and the generator then learns to ignore them. In that case RAG degrades to plain BART.

RAG is also more expensive at inference than a closed-book model, because it must retrieve and then run the generator once per document. Finally, FEVER results show that RAG is still a few points behind heavily engineered pipelines when strong evidence supervision is available, so end-to-end learning without document labels is convenient but not always optimal.

## Fun trivia and history (from your web search)

The three letters RAG have become one of the most famous acronyms in AI product language. Before this paper, people said retrieve-and-read or open-domain QA. After this paper, startups, cloud providers, and open-source libraries all advertise RAG pipelines, RAG evaluation, and RAG chat over your docs. The paper's plain descriptive title therefore turned into an industry category, much like the way the word transformer outgrew its original paper.

A second memorable detail is the hot-swap experiment with world leaders. Instead of only reporting accuracy numbers, the authors built two Wikipedia indexes, one from 2016 and one from 2018, and asked who holds each leadership position. With the matching index, accuracy was around 70 percent for 2016 leaders and 68 percent for 2018 leaders. With the mismatched index, accuracy collapsed to 12 percent and 4 percent. That simple before-and-after demo made the abstract idea of updatable knowledge feel concrete. It also foreshadowed today's practice of updating a chatbot by re-indexing documents instead of retraining the model. The authors even shipped an early interactive demo through HuggingFace Transformers, which helped the method spread quickly beyond the lab.

## Key terms explained (glossary of 5-8 terms)

Parametric memory means knowledge stored in the weights of a neural network. It is like what you remember in your head after studying. It is powerful but hard to edit.

Non-parametric memory means knowledge stored outside the weights as searchable text. It is like books on a shelf. It is easy to read, inspect, and replace.

Dense Passage Retrieval, or DPR, is a retriever that uses two BERT encoders to map questions and passages into vectors. Passages with high dot-product similarity to the question are returned. It finds meaning matches, not just word matches.

Maximum Inner Product Search, or MIPS, is the fast search algorithm that finds the top-scoring vectors in a huge index without comparing against every entry one by one. FAISS is the library the paper uses to do this.

Marginalization over documents means averaging the generator's predictions across several retrieved passages, weighted by how likely each passage seemed. It is a principled way of saying that the model is not sure which source is best, so it listens to all of them.

RAG-Sequence means one passage is held responsible for the whole answer. RAG-Token means different passages can be responsible for different words. The first is simpler, while the second can fuse facts from multiple places.

Exact Match, or EM, is a strict question-answering metric. The generated answer must equal the gold answer after normalization. There is no partial credit.

Hallucination means fluent but false generation. RAG reduces hallucination by forcing the generator to look at retrieved evidence, though it does not eliminate the problem entirely.

## Connections (which earlier paper it builds on, which later paper builds on it)

This paper builds directly on the Transformer and BERT lineage. Paper 01, Attention Is All You Need, provides the sequence-to-sequence architecture. Paper 02, BERT, provides the pre-trained encoders used in DPR and the general idea of pre-training before fine-tuning. Paper 03, GPT-3, provides the closed-book baseline that RAG is trying to improve, especially the finding that very large models store facts but still hallucinate and go stale. It also builds on BART for generation and on DPR, REALM, and ORQA for learned retrieval.

Later work builds on it across the whole set. Papers 08 (InstructGPT), 13 (LLaMA 2), and 15 (DeepSeek-R1) make assistants more helpful and better at reasoning, but they all benefit from grounding in retrieved documents when factual freshness matters. Paper 11 (LoRA) makes fine-tuning the generator and query encoder cheaper, which is exactly the fine-tuning step RAG requires. In the broader field, Fusion-in-Decoder, Atlas, RETRO, and every modern enterprise search-plus-chat stack are direct intellectual children of the RAG-Sequence and RAG-Token distinction.
