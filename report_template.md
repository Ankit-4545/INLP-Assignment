# Assignment 1: Tokenization and Language Modeling
**Introduction to NLP - Spring '26**

**Name:** [Your Name]  
**Roll Number:** [Your Roll Number]  
**Date:** [Submission Date]

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
- Punctuation remains attached to adjacent words
- No special handling of contractions (e.g., "don't" stays as one token)

#### Regex Tokenizer
**Description:** Uses regular expression to separate words, numbers, and punctuation.

**Pattern:** `[^\W\d_]+|\d+|[^\w\s]`

**Simplifying Assumptions:**
- Each punctuation mark is treated as a separate token
- Numbers are kept as single tokens
- Unicode word characters supported

#### BPE Tokenizer
**Description:** Byte Pair Encoding learns subword units through iterative merging.

**Parameters:** 1500 merge operations

**Simplifying Assumptions:**
- Words initially split by whitespace
- End-of-word marker `</w>` added to each word
- Character-level fallback for unknown words

### 2.3 Tokenization Analysis (Task 1.3)

#### English Corpus

##### Sensible Tokenizations (3 examples)

**Example 1:**
- **Sentence:** [Insert actual sentence from your output]
- **Tokenizer:** [Which tokenizer - Whitespace/Regex/BPE]
- **Output:** `[Show the actual tokens]`
- **Analysis:** [Explain why this tokenization is sensible for language modeling. Consider: Does it preserve meaning? Does it help predict next words? Does it handle morphology well?]

**Example 2:**
- **Sentence:** [Another sentence]
- **Tokenizer:** [Which tokenizer]
- **Output:** `[Tokens]`
- **Analysis:** [Your reasoning]

**Example 3:**
- **Sentence:** [Another sentence]
- **Tokenizer:** [Which tokenizer]
- **Output:** `[Tokens]`
- **Analysis:** [Your reasoning]

##### Non-Sensible Tokenizations (3 examples)

**Example 1:**
- **Sentence:** [Problematic sentence]
- **Tokenizer:** [Which tokenizer]
- **Output:** `[Tokens that show the problem]`
- **Problem:** [Explain what went wrong. Does it break meaningful units? Create too many/few tokens? Handle punctuation poorly?]

**Example 2:**
- **Sentence:** [Another problematic case]
- **Tokenizer:** [Which tokenizer]
- **Output:** `[Tokens]`
- **Problem:** [Explain the issue]

**Example 3:**
- **Sentence:** [Another problematic case]
- **Tokenizer:** [Which tokenizer]
- **Output:** `[Tokens]`
- **Problem:** [Explain the issue]

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
| Whitespace | [Value]      | [Value]     | [Value]    |
| Regex      | [Value]      | [Value]     | [Value]    |
| BPE        | [Value]      | [Value]     | [Value]    |

*Note: Fill in actual values from your program output*

#### Analysis of Perplexity Results:

**Best Overall Model:**
- [Which tokenizer + smoothing combination achieved lowest perplexity?]
- [What was the perplexity value?]

**Effect of Smoothing:**
- **No smoothing:** [What happened? Infinite perplexity? Why?]
- **Witten-Bell:** [Performance? How does it compare?]
- **Kneser-Ney:** [Performance? Why might it be better/worse?]

**Effect of Tokenization:**
- **Whitespace:** [How did it perform across smoothing methods?]
- **Regex:** [How did it perform? Better or worse than whitespace?]
- **BPE:** [How did it perform? Why might subword units help/hurt?]

**Key Observations:**
- [Any patterns you notice?]
- [Trade-offs between vocabulary size and perplexity?]
- [Why does smoothing matter so much?]

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
- **Original Sentence:** [Full test sentence]
- **Prefix Given:** [Prefix]
- **Model Used:** [Model]
- **Model Output:** [Generation]
- **Problem:** [What's wrong?]
- **Why It Happened:** [Explanation]

#### Behavioral Observations

##### Effect of Smoothing on Generation Quality

**No Smoothing:**
- [How does MLE-only model behave? Does it fail often? Get stuck?]

**Witten-Bell:**
- [Does it generate more diverse text? More errors? Better/worse coherence?]

**Kneser-Ney:**
- [How does generation quality compare? Better word choices?]

##### Effect of Tokenization on Generation Quality

**Whitespace:**
- [Does it handle punctuation well? Generate natural text?]
- [Any issues with compound words or contractions?]

**Regex:**
- [Better punctuation handling?]
- [More grammatical output?]
- [Any new problems introduced?]

**BPE:**
- [Does subword tokenization help or hurt readability?]
- [Can it generate rare words better?]
- [Any artifacts like "</w>" in output?]

##### Common Failure Modes Across Models

1. [Describe a common problem - e.g., repetition, nonsense, getting stuck]
2. [Another common issue]
3. [Another pattern you noticed]

---

## 4. Conclusion

### 4.1 Key Findings

**Tokenization:**
- For **English**: [Best tokenizer and why]
- For **Mongolian**: [Best tokenizer and why]
- **Trade-offs**: [Vocabulary size vs. meaningful units vs. handling unknowns]

**Language Modeling:**
- **Best smoothing method**: [Which one and why]
- **Best tokenizer for LM**: [Which one and why]
- **Perplexity vs. generation quality**: [Do they always correlate?]

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

## Appendix: Code Examples

[Optional: Include interesting code snippets or additional analysis]
