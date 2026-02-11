# Assignment 1: Tokenization and Language Modeling
**Introduction to NLP - Spring '26**

**Name:** [Ankit Kumar]  
**Roll Number:** [2025201022]  
**Date:** [11/02/2026]

---

## 1. Introduction

This report presents the implementation and analysis of tokenization methods and N-gram language models with various smoothing techniques for English and Mongolian corpora from the CC-100 dataset.

---

## 2. Task 1: Tokenization

### 2.1 Data Cleaning and Partitioning (Task 1.1)

#### Cleaning Steps Implemented:
- Removal of URLs using regex pattern
- Normalization of Unicode quotes and apostrophes
- Removal of non-breaking spaces
- Collapsing multiple whitespace characters
- Stripping leading/trailing whitespace

#### Data Partitioning Ratios:
- **Training:** 80%
- **Validation:** 10%
- **Testing:** 10%

### 2.2 Tokenizer Implementation (Task 1.2)

#### Whitespace Tokenizer
**Description:** Splits text on whitespace characters only.

**Simplifying Assumptions:**
- Punctuation remains attached to adjacent words if there is no space between them.
- No special handling of contractions (e.g., "don't" stays as one token).

#### Regex Tokenizer
**Description:** Uses regular expression to separate words, numbers, and punctuation.

**Pattern:** `[^\W\d_]+|\d+|[^\w\s]`

**Simplifying Assumptions:**
- Each punctuation mark is treated as a separate token
- Numbers are kept as single tokens
- Unicode word characters supported

#### BPE Tokenizer
**Description:** Byte Pair Encoding learns subword units through iterative merging.

**Parameters:** 1000 merge operations

**Simplifying Assumptions:**
- Words initially split by whitespace
- End-of-word marker `</w>` added to each word
- Character-level fallback for unknown words

### 2.3 Tokenization Analysis (Task 1.3)

#### Sensible Tokenizations English corpus (3 examples)

##### Whitespace Tokenizer

**Example 1:**
- **Sentence:** 'The cat sat'
- **Output:** `['The', 'cat', 'sat']`
- **Analysis:** This tokenization preserves complete words and semantic meaning, allowing the model to learn strong word-level associations such as “cat → sat.” It supports reliable next-word prediction because these common word sequences frequently appear in training data

**Example 2:**
- **Sentence:** 'Hello, world!'
- **Output:** `['Hello,', 'world!']`
- **Analysis:** Although punctuation remains attached, the core words are preserved, enabling the model to capture meaningful co-occurrence patterns. It is still useful for predicting sentence-level structures like greetings despite minor vocabulary inflation.

**Example 3:**
- **Sentence:** 'I love NLP'
- **Output:** `['I', 'love', 'NLP']`
- **Analysis:** Each token represents a clear linguistic unit, making probability estimation straightforward. Frequent phrases like “I love” help the model generalize well and improve prediction accuracy.

##### Regex Tokenizer

**Example 1:**
- **Sentence:** 'hello!'
- **Output:** `['hello', '!']`
- **Analysis:** Separating punctuation helps the model explicitly learn that exclamation marks often follow greetings. This improves sentence boundary prediction and reduces noise caused by punctuation-attached words.

**Example 2:**
- **Sentence:** 'Price is $5'
- **Output:** `['price', 'is', '$', '5']`
- **Analysis:** Breaking currency symbols from numbers allows the model to learn structured patterns in financial expressions. It improves generalization since “$” and numeric values can appear in multiple contexts.

**Example 3:**
- **Sentence:** 'Are you sure?'
- **Output:** `['are', 'you', 'sure', '?']`
- **Analysis:** The tokenizer preserves word meaning while isolating the question mark, helping the model recognize interrogative sentence patterns. This supports better prediction of conversational structures.

##### BPE Tokenizer

**Example 1:**
- **Sentence:** 'running'
- **Output:** `['run', 'ing']`
- **Analysis:** Splitting into “run” and “ing” captures morphological structure, allowing the model to generalize across related forms like “walking” or “playing.” This reduces sparsity and improves prediction reliability.

**Example 2:**
- **Sentence:** 'unhappiness'
- **Output:** `['un', 'happi', 'ness']`
- **Analysis:** The subwords represent prefix, root, and suffix components, enabling the model to understand negation and noun formation. Such decomposition improves handling of rare or unseen words.

**Example 3:**
- **Sentence:** 'International'
- **Output:** `['inter', 'national']`
- **Analysis:** Dividing the word into frequent subunits reduces vocabulary size while preserving semantic cues. These reusable components strengthen probability estimates across different contexts.

#### Non-Sensible Tokenizations English corpus (3 examples)

##### Whitespace Tokenizer

**Example 1:**
- **Sentence:** 'hello!'
- **Output:** `['hello!']`
- **Problem:** Attaching punctuation to the word creates a new token that rarely appears elsewhere, increasing vocabulary size and sparsity while weakening generalization.

**Example 2:**
- **Sentence:** [hello ,  world!]
- **Output:** `['hello', ',', 'world!']`
- **Problem:** [Explain the issue]

**Example 3:**
- **Sentence:** [don't stop]
- **Output:** `['don't', 'stop']`
- **Problem:** Keeping the contraction as a single token prevents the model from learning relationships between “do” and “not,” reducing its ability to generalize across similar constructions.

##### Regex Tokenizer

**Example 1:**
- **Sentence:** 'U.S.A.'
- **Output:** `['u', '.', 's', '.', 'a']`
- **Problem:** Splitting an abbreviation into individual letters destroys its semantic identity, producing tokens that rarely occur independently and harming probability estimation.

**Example 2:**
- **Sentence:** 'C++'
- **Output:** `['c', '+', '+']`
- **Problem:** Breaking a technical term into symbols fragments a meaningful unit, making it harder for the model to learn domain-specific vocabulary.

**Example 3:**
- **Sentence:** '3.14'
- **Output:** `['3', '.', '14']`
- **Problem:** Separating a decimal number disrupts a single numeric concept, potentially confusing the model when learning patterns involving measurements or prices.

##### BPE Tokenizer

**Example 1:**
- **Sentence:** 'the'
- **Output:** `['t', 'he']`
- **Problem:** Over-segmentation of a very frequent word increases sequence length unnecessarily and weakens fluency during generation.

**Example 2:**
- **Sentence:** 'market'
- **Output:** `['mar', 'ke', 't']`
- **Problem:** Fragmenting a common word into multiple subwords reduces interpretability and may dilute meaningful statistical patterns.

**Example 3:**
- **Sentence:** '$63'
- **Output:** `['$' '6', '3']`
- **Problem:** Splitting the currency amount into small units breaks a coherent numerical expression, making it harder for the model to learn financial formats.

#### Mongolian Corpus

##### Sensible Tokenizations (3 examples)

[Same structure as English - use actual Mongolian examples from your output]

##### Non-Sensible Tokenizations (3 examples)

[Same structure as English]

##### Special Considerations for Mongolian:
- Mongolian uses Cyrillic script (different character properties than Latin)
- Agglutinative morphology (words formed by adding affixes)
- How does BPE handle morphological boundaries?
- Does regex tokenization work well with Cyrillic punctuation?

#### Summary: Which Tokenizer for Which Language?

**English:**
- **Best tokenizer:** [Your choice based on analysis]
- **Reasoning:** [Why? Consider vocabulary size, handling of morphology, punctuation separation, etc.]

**Mongolian:**
- **Best tokenizer:** [Your choice]
- **Reasoning:** [Why? Consider agglutinative morphology, Cyrillic script properties, etc.]

---

## 3. Task 2: Language Modeling

### 3.1 Implementation Details (Task 2.1, 2.2)

#### Model Configuration
- **N-gram order:** 4 (4-grams)
- **Smoothing methods:** None (MLE baseline), Witten-Bell, Kneser-Ney
- **Kneser-Ney discount:** 0.75

#### Simplifying Assumptions
- **Greedy decoding:** Most likely token selected at each step (not beam search)
- **Sentence markers:** Padded with `<s>` (start) and `</s>` (end)
- **Generation termination:** Stops at `</s>` or after 50 tokens

### 3.2 Perplexity Results (Task 2.3)

| Tokenizer  | No Smoothing | Witten-Bell | Kneser-Ney |
|------------|--------------|-------------|------------|
| Whitespace |    [inf]     | [89858.96]  | [2240.72]  |
| Regex      |    [inf]     | [14371.2]   | [549.39]   |
| BPE        |    [inf]     | [64.86]     | [44.26]    |

#### Analysis of Perplexity Results:

**Best Overall Model:**
- The combination of BPE tokenizer and Kneser–Ney smoothing gave the lowest perplexity.
- With a value of 44.26, it provided the most accurate and consistent predictions across all models.

**Effect of Smoothing:**
- **No smoothing:** All tokenizers resulted in infinite perplexity because unseen n-grams were assigned zero probability, making the model unable to generalize to new sequences.

- **Witten-Bell:** Significantly reduced perplexity compared to no smoothing by redistributing probability mass to unseen events, though performance varied depending on tokenization.

- **Kneser-Ney:** Delivered the best performance as it considers context diversity rather than relying only on frequency, leading to more accurate probability estimates.

**Effect of Tokenization:**
- **Whitespace:** Performed the worst due to very large vocabulary and severe sparsity, resulting in extremely high perplexity even with smoothing.

- **Regex:** Improved over whitespace by producing cleaner tokens and slightly reducing vocabulary size, but still suffered from sparsity inherent to word-level tokenization.

- **BPE:** Achieved dramatically lower perplexity because subword units reduce vocabulary size, improve coverage of rare words, and provide more reliable n-gram counts.

**Key Observations:**
- A clear pattern shows that reducing vocabulary size leads to lower perplexity by minimizing sparse contexts.
- There is a strong trade-off between vocabulary size and predictive confidence, with subword tokenization offering better statistical efficiency.
- Smoothing is critical in language modeling because it prevents zero probabilities and enables the model to handle unseen sequences effectively.

### 3.3 Qualitative Analysis (Task 2.4)

#### Correct/Sensible Completions (3 examples)

**Example 1:**
- **Original Sentence:** [Full test sentence]
- **Prefix Given:** [First half of sentence]
- **Model Used:** [Tokenizer + Smoothing method]
- **Model Output:** [What the model generated]
- **Analysis:** [Why is this good? Does it complete the sentence sensibly? Is it grammatical? Does it match the original meaning?]

**Example 2:**
- **Original Sentence:** [Full test sentence]
- **Prefix Given:** [Prefix]
- **Model Used:** [Model]
- **Model Output:** [Generation]
- **Analysis:** [Your analysis]

**Example 3:**
- **Original Sentence:** [Full test sentence]
- **Prefix Given:** [Prefix]
- **Model Used:** [Model]
- **Model Output:** [Generation]
- **Analysis:** [Your analysis]

#### Incorrect/Problematic Completions (3 examples)

**Example 1:**
- **Original Sentence:** [Full test sentence]
- **Prefix Given:** [Prefix]
- **Model Used:** [Tokenizer + Smoothing]
- **Model Output:** [What the model generated]
- **Problem:** [What went wrong? Ungrammatical? Nonsensical? Repetitive? Wrong topic?]
- **Why It Happened:** [Explain based on model limitations - sparse data? Greedy decoding? Tokenization issue?]

**Example 2:**
- **Original Sentence:** [Full test sentence]
- **Prefix Given:** [Prefix]
- **Model Used:** [Model]
- **Model Output:** [Generation]
- **Problem:** [What's wrong?]
- **Why It Happened:** [Explanation]

**Example 3:**
- **Original Sentence:** [Full test sentence]e
- **Prefix Given:** [Prefix]
- **Model Used:** [Model]
- **Model Output:** [Generation]
- **Problem:** [What's wrong?]
- **Why It Happened:** [Explanation]

#### Behavioral Observations

##### Effect of Smoothing on Generation Quality

**No Smoothing:**
- Without smoothing, unseen n-grams receive zero probability, making the model highly rigid and prone to repetition. Text generation often resembles memorized training data and may terminate when an unseen context appears. While it can produce grammatically correct sentences on small datasets, it generally fails to generalize as data grows.

**Witten-Bell:**
- Witten–Bell redistributes probability mass to unseen events based on the number of unique continuations for a context, enabling more flexible text generation. It reduces repetition and prevents dead ends but may sometimes allow unlikely word combinations. Overall, it provides a balance between determinism and creativity.

**Kneser-Ney:**
- Kneser–Ney improves generation quality by considering how widely a word appears across different contexts rather than relying only on frequency. This leads to more natural transitions, better fluency, and fewer unnatural phrases. It is widely regarded as the most effective smoothing method for higher-order n-gram models.

##### Effect of Tokenization on Generation Quality

Tokenization plays a crucial role in language modeling because it defines the unit on which probabilities are learned. The choice of tokenizer directly impacts vocabulary size, data sparsity, and the model’s ability to generalize, all of which influence the fluency and coherence of generated text.

**Whitespace:**
- Splits text only on spaces, resulting in a very large vocabulary.
- Causes severe data sparsity, making higher-order n-gram predictions unreliable.
- Generated text is often repetitive, rigid, and closely resembles memorized training data.

**Regex:**
- Separates punctuation and symbols, producing cleaner and more consistent tokens.
- Slightly improves probability estimation but still suffers from vocabulary explosion.
- Text generation is somewhat more structured than whitespace but remains limited by rare words.

**BPE:**
- Breaks words into frequent subword units, greatly reducing vocabulary size.
- Minimizes sparsity and improves the model’s ability to handle unseen words.
- Produces more fluent, flexible, and natural-sounding generated text.

##### Common Failure Modes Across Models

1. [Describe a common problem - e.g., repetition, nonsense, getting stuck]
2. [Another common issue]
3. [Another pattern you noticed]

---

## 4. Conclusion

### 4.1 Key Findings

**Tokenization:**
- For English **Byte Pair Encoding (BPE)** performed best because it reduced vocabulary size while preserving meaningful subword patterns, leading to better generalization and improved handling of unseen words.
- For Mongolian **BPE** was most effective due to the language’s rich morphology, where many word forms are created through suffixes. Subword tokenization helped capture these patterns and significantly reduced sparsity.
- **Trade-offs**: Tokenization requires balancing vocabulary size, meaningful representation, and handling of unknown words. Word-level methods preserve full semantic units but create large vocabularies and struggle with unseen terms. In contrast, BPE reduces vocabulary size and handles unknown words effectively through subwords, though tokens may be slightly less interpretable.

**Language Modeling:**
- **Best smoothing method**: Kneser–Ney smoothing performed best because it considers the diversity of contexts in which words appear, producing more reliable probability estimates and more natural text generation.

- **Best tokenizer for LM**: BPE proved most suitable for language modeling as it minimizes sparsity, stabilizes probability distributions, and improves both perplexity and generation fluency.

- **Perplexity vs. generation quality**: Lower perplexity generally indicates better predictive performance, but it does not always guarantee superior text generation. Generation quality depends on how probability mass is distributed; overly confident models may produce repetitive text, while well-balanced models generate more natural and diverse language.

### 4.2 Lessons Learned

1. **Importance of smoothing**: [Why is it critical for n-gram models?]
2. **Tokenization matters**: [How much does choice of tokenizer affect results?]
3. **Greedy decoding limitations**: [What are the downsides?]
4. **Data sparsity**: [How does 4-gram sparsity affect performance?]
5. **Subword units**: [When are they helpful vs. harmful?]

### 4.3 Potential Improvements

If you had more time/resources, what would you try?
- Larger training corpus?
- Different n-gram orders?
- Beam search instead of greedy?
- Different BPE merge counts?
- Other smoothing methods?

---

## 5. References

1. CS224N – Language Modelling and Smoothing
2. Bill McCartney – NLP Lunch Tutorial: Smoothing  
3. Jurafsky & Martin Chapter 3 – N-gram Language Models
4. Jurafsky & Martin Appendix B – Kneser-Ney Smoothing
5. Wikipedia – Mongolian Language

---