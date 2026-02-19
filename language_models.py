import math
from collections import defaultdict, Counter
from tokenizers import (
        WhitespaceTokenizer, 
        RegexTokenizer, 
        BPETokenizer, 
        clean_corpus, 
        split_corpus
    )

class NGramLanguageModel:
    """Base n-gram language model"""
    
    def __init__(self, n=4):
        
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
    
    def train(self, tokenized_corpus):
        
        for sentence in tokenized_corpus:
            # Add start and end tokens
            padded = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            
            # Count n-grams
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                self.vocab.update(padded)
    
    # Get probability of an n-gram (to be overridden by smoothing methods)
    def get_probability(self, ngram):

        context = ngram[:-1]
        
        if self.context_counts[context] == 0:
            return 0.0
        
        # Basic MLE without smoothing
        return self.ngram_counts[ngram] / self.context_counts[context]
    
    # Calculate perplexity on test set
    def perplexity(self, test_sentences):
        
        log_prob_sum = 0
        word_count = 0
        
        for sentence in test_sentences:
            padded = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                prob = self.get_probability(ngram)
                
                if prob > 0:
                    log_prob_sum += math.log(prob)
                else:
                    # Handle zero probability
                    log_prob_sum += -float('inf')
                
                word_count += 1
        
        # Perplexity = exp(-1/N * sum(log p(w_i)))
        avg_log_prob = log_prob_sum / word_count if word_count > 0 else 0
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    # Predict next token given context
    def predict_next_token(self, context):
    
        # Get all possible next tokens
        candidates = {}
        for ngram in self.ngram_counts:
            if ngram[:-1] == context:
                next_token = ngram[-1]
                candidates[next_token] = self.get_probability(ngram)
        
        if not candidates:
            # Backoff to shorter context
            if len(context) > 1:
                return self.predict_next_token(context[1:])
            return '</s>'  # End sentence if nothing found
        
        # Return token with highest probability
        return max(candidates, key=candidates.get)
    
    # Complete a sentence given partial input
    def complete_sentence(self, partial_sentence, max_length=50):
    
        result = partial_sentence.copy()
        
        # Pad context if needed
        context = ['<s>'] * (self.n - 1) + result
        context = tuple(context[-(self.n - 1):])
        
        for _ in range(max_length):
            next_token = self.predict_next_token(context)
            
            if next_token == '</s>':
                break
            
            result.append(next_token)
            # Update context
            context = tuple(list(context[1:]) + [next_token])
        
        return result

# Witten-Bell smoothing implementations
class WittenBellLM(NGramLanguageModel):
    
    def __init__(self, n=4):
        super().__init__(n)
        self.num_unique_continuations = defaultdict(int)
    
    def train(self, tokenized_corpus):
        """Train and compute unique continuation counts"""
        super().train(tokenized_corpus)
        
        # Count number of unique words that follow each context
        continuation_sets = defaultdict(set)
        for ngram in self.ngram_counts:
            context = ngram[:-1]
            word = ngram[-1]
            continuation_sets[context].add(word)
        
        for context, words in continuation_sets.items():
            self.num_unique_continuations[context] = len(words)
    
    def get_probability(self, ngram):
        """
        Witten-Bell smoothing formula:
        P_WB(w|context) = (C(context, w) + T(context) * P_backoff(w)) / (C(context) + T(context))
        where T(context) is number of unique word types following context
        
        This uses interpolation: seen n-grams get probability proportional to their count,
        and the remaining probability mass is distributed according to backoff.
        """
        context = ngram[:-1]
        word = ngram[-1]
        
        count_ngram = self.ngram_counts[ngram]
        count_context = self.context_counts[context]
        T = self.num_unique_continuations[context]
        
        if count_context == 0:
            # Unseen context: use uniform distribution over vocabulary
            return 1.0 / len(self.vocab) if len(self.vocab) > 0 else 0.0
        lambda_wb = T / (count_context + T)
        
        # Backoff probability (uniform over vocabulary for simplicity)
        p_backoff = 1.0 / len(self.vocab) if len(self.vocab) > 0 else 0.0
        if count_ngram > 0:
            p_ml = count_ngram / count_context
            prob = (1 - lambda_wb) * p_ml + lambda_wb * p_backoff
        else:
            # Unseen word in this context: only get backoff probability
            prob = lambda_wb * p_backoff
        
        return prob


# Kneser-Ney smoothing implementation
class KneserNeyLM(NGramLanguageModel):
    
    def __init__(self, n=4, discount=0.75):
        super().__init__(n)
        self.discount = discount
        self.continuation_counts = defaultdict(int)
    
    def train(self, tokenized_corpus):
        """Train and compute continuation counts for KN"""
        super().train(tokenized_corpus)
    
        unique_contexts = defaultdict(set)
        for ngram in self.ngram_counts:
            word = ngram[-1]
            context = ngram[:-1]
            unique_contexts[word].add(context)
        
        for word, contexts in unique_contexts.items():
            self.continuation_counts[word] = len(contexts)
        
        # Also compute unique continuations for lambda calculation
        self.num_unique_continuations = defaultdict(int)
        continuation_sets = defaultdict(set)
        for ngram in self.ngram_counts:
            context = ngram[:-1]
            word = ngram[-1]
            continuation_sets[context].add(word)
        
        for context, words in continuation_sets.items():
            self.num_unique_continuations[context] = len(words)
    
    def get_probability(self, ngram):
        """
        Kneser-Ney smoothing:
        P_KN(w|context) = max(count(context,w) - D, 0) / count(context) + 
                          lambda(context) * P_continuation(w)
        
        where:
        - D is the discount (typically 0.75)
        - lambda(context) = D * |{w : count(context, w) > 0}| / count(context)
        - P_continuation(w) = |{context : count(context, w) > 0}| / |all bigram types|
        
        For unseen words (OOV), we interpolate with a uniform distribution.
        """
        context = ngram[:-1]
        word = ngram[-1]
        
        count_ngram = self.ngram_counts[ngram]
        count_context = self.context_counts[context]
        
        # Uniform probability for OOV handling
        vocab_size = len(self.vocab) if len(self.vocab) > 0 else 1
        p_uniform = 1.0 / vocab_size
        
        # Small interpolation weight for uniform distribution (handles OOV)
        epsilon = 0.01
        
        if count_context == 0:
            # Backoff to continuation probability for unseen context
            total_continuation = sum(self.continuation_counts.values())
            if total_continuation > 0:
                p_cont = self.continuation_counts.get(word, 0) / total_continuation
                # Interpolate with uniform for OOV
                return (1 - epsilon) * p_cont + epsilon * p_uniform
            return p_uniform
        
        # First term: discounted probability
        discounted_prob = max(count_ngram - self.discount, 0) / count_context
        
        # Lambda: normalization factor for backoff probability mass
        num_unique_following = self.num_unique_continuations.get(context, 0)
        lambda_factor = (self.discount * num_unique_following) / count_context
        
        # Continuation probability: how many contexts does this word appear in?
        total_continuation = sum(self.continuation_counts.values())
        if total_continuation > 0:
            p_continuation = self.continuation_counts.get(word, 0) / total_continuation
        else:
            p_continuation = p_uniform
        
        # Final interpolated probability with OOV handling
        prob = discounted_prob + lambda_factor * p_continuation
        
        # Interpolate with uniform for robustness (handles completely unseen words)
        prob = (1 - epsilon) * prob + epsilon * p_uniform
        
        return prob


def load_jsonl_corpus(filepath, max_lines=5000):

    import json
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            try:
                data = json.loads(line.strip())
                if 'text' in data and data['text'].strip():
                    texts.append(data['text'].strip())
            except json.JSONDecodeError:
                continue
    return texts


def run_full_experiment(corpus_path, language_name, max_lines=3000):
    
    
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT FOR {language_name.upper()}")
    print(f"{'='*70}")
    
    # Load corpus
    print(f"\n[1] Loading corpus from {corpus_path}...")
    raw_texts = load_jsonl_corpus(corpus_path, max_lines=max_lines)
    
    # Clean corpus
    print("\n[2] Cleaning corpus...")
    cleaned_texts = []
    for text in raw_texts:
        cleaned = clean_corpus(text)
        if cleaned and len(cleaned) > 10:  # Skip very short texts
            cleaned_texts.append(cleaned)
    
    # Initialize tokenizers
    print("\n[3] Initializing tokenizers...")
    tokenizers = {
        'Whitespace': WhitespaceTokenizer(),
        'Regex': RegexTokenizer(),
        'BPE': BPETokenizer(num_merges=1000)
    }
    
    # Train tokenizers and prepare data splits
    print("\n[4] Training tokenizers and preparing data splits...")
    tokenized_data = {}
    
    for name, tokenizer in tokenizers.items():
        print(f"\n    Training {name} tokenizer...")
        tokenizer.train(cleaned_texts)
        
        # Tokenize all sentences
        tokenized = [tokenizer.tokenize(text) for text in cleaned_texts]
        # Filter out empty tokenizations
        tokenized = [t for t in tokenized if len(t) > 0]
        
        # Split into train/val/test
        train, val, test = split_corpus(tokenized, train_ratio=0.8, val_ratio=0.1)
        tokenized_data[name] = {
            'train': train,
            'val': val,
            'test': test,
            'vocab_size': len(tokenizer.vocab)
        }
        
        print(f"    {name}: Vocab={len(tokenizer.vocab)}, Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        # Show sample tokenization
        if cleaned_texts:
            sample = cleaned_texts[0]
            sample_tokens = tokenizer.tokenize(sample)
            print(f"    Sample: '{sample}'")
            print(f"    Tokens: {sample_tokens}")
    results = {}
    if language_name == "Mongolian":
        return results, tokenized_data
    # Train language models and evaluate
    print(f"\n[5] Training language models and calculating perplexity...")
    print("\n" + "-"*70)
    print(f"{'Tokenizer':<15} {'Smoothing':<15} {'Perplexity':<15} {'Train Time'}")
    print("-"*70)
    
    
    import time
    
    for tok_name, data in tokenized_data.items():
        results[tok_name] = {}
        
        models = {
            'No Smoothing': NGramLanguageModel(n=4),
            'Witten-Bell': WittenBellLM(n=4),
            'Kneser-Ney': KneserNeyLM(n=4, discount=0.75)
        }
        
        for model_name, model in models.items():
            start_time = time.time()
            
            # Train
            model.train(data['train'])
            train_time = time.time() - start_time
            
            # Calculate perplexity on test set
            try:
                perplexity = model.perplexity(data['test'])
                if math.isinf(perplexity):
                    perp_str = "inf (OOV)"
                else:
                    perp_str = f"{perplexity:.2f}"
            except Exception as e:
                perp_str = f"Error: {str(e)[:20]}"
                perplexity = float('inf')
            
            results[tok_name][model_name] = {
                'model': model,
                'perplexity': perplexity
            }
            
            print(f"{tok_name:<15} {model_name:<15} {perp_str:<15} {train_time:.2f}s")
    
    # Demonstrate autocomplete
    print(f"\n[6] Autocomplete Demonstrations for {language_name}")
    print("-"*70)
    
    # For each tokenizer, get prompts from ITS OWN training data
    for tok_name in tokenizers.keys():
        print(f"\n  {tok_name} Tokenizer:")
        
        # Get prompts from this tokenizer's training data
        tok_prompts = []
        data = tokenized_data[tok_name]
        if data['train']:
            for sent in data['train'][:100]:
                if len(sent) >= 3:
                    tok_prompts.append(sent[:3])
                    if len(tok_prompts) >= 2: 
                        break
        
        if not tok_prompts:
            tok_prompts = [['the']]  # Fallback
        
        for model_name in ['No Smoothing', 'Witten-Bell', 'Kneser-Ney']:
            model = results[tok_name][model_name]['model']
            print(f"    {model_name}:")
            for prompt in tok_prompts:
                try:
                    completed = model.complete_sentence(prompt.copy(), max_length=10)
                    print(f"      '{' '.join(prompt)}' -> '{' '.join(completed)}'")
                except Exception as e:
                    print(f"      '{' '.join(prompt)}' -> Error: {e}")
    
    return results, tokenized_data


def print_qualitative_analysis(results, tokenized_data, language_name):
    print(f"\n{'='*70}")
    print(f"QUALITATIVE ANALYSIS FOR {language_name.upper()}")
    print(f"{'='*70}")
    
    # Get actual examples from the data
    print("\n--- Sample Tokenization Comparisons ---")
    for tok_name, data in tokenized_data.items():
        if data['train']:
            sample = data['train'][0][:8]
            print(f"  {tok_name}: {sample}")
    
    if language_name == "Mongolian":
        return
    
    # Print perplexity summary with analysis
    print("\n--- Perplexity Summary with Interpretation ---")
    best_perp = float('inf')
    best_config = ""
    
    for tok_name, models in results.items():
        for model_name, data in models.items():
            perp = data['perplexity']
            if math.isinf(perp):
                print(f"  {tok_name} + {model_name}: inf (OOV - zero probability assigned)")
            else:
                print(f"  {tok_name} + {model_name}: {perp:.2f}")
                if perp < best_perp:
                    best_perp = perp
                    best_config = f"{tok_name} + {model_name}"
    
    print(f"\n  Best configuration: {best_config} (perplexity: {best_perp:.2f})")


# Example usage
if __name__ == "__main__":
    import os
    
    # Paths to datasets
    english_path = "cc100_en.jsonl"  # Update with actual path
    mongolian_path = "cc100_mn.jsonl"  # Update with actual path

    # Check if datasets exist
    if not os.path.exists(english_path):
        print(f"WARNING: English dataset not found at {english_path}")
    if not os.path.exists(mongolian_path):
        print(f"WARNING: Mongolian dataset not found at {mongolian_path}")
    
    # Run experiments
    all_results = {}
    
    # English experiment
    if os.path.exists(english_path):
        en_results, en_data = run_full_experiment(
            english_path, 
            "English", 
            max_lines=2000  # Adjust based on available memory
        )
        all_results['English'] = en_results
        print_qualitative_analysis(en_results, en_data, "English")
    
    # Mongolian experiment
    if os.path.exists(mongolian_path):
        mn_results, mn_data = run_full_experiment(
            mongolian_path, 
            "Mongolian", 
            max_lines=2000
        )
        all_results['Mongolian'] = mn_results
        print_qualitative_analysis(mn_results, mn_data, "Mongolian")
    
    # Final summary
    print("FINAL SUMMARY")
    
    print("\nPerplexity Comparison Table:")
    print("-"*70)
    print(f"{'Language':<12} {'Tokenizer':<12} {'No Smooth':<12} {'Witten-Bell':<12} {'Kneser-Ney':<12}")
    print("-"*70)
    
    for lang, results in all_results.items():
        for tok_name in results.keys():
            no_smooth = results[tok_name]['No Smoothing']['perplexity']
            wb = results[tok_name]['Witten-Bell']['perplexity']
            kn = results[tok_name]['Kneser-Ney']['perplexity']
            
            ns_str = "inf" if math.isinf(no_smooth) else f"{no_smooth:.1f}"
            wb_str = "inf" if math.isinf(wb) else f"{wb:.1f}"
            kn_str = "inf" if math.isinf(kn) else f"{kn:.1f}"
            
            print(f"{lang:<12} {tok_name:<12} {ns_str:<12} {wb_str:<12} {kn_str:<12}")
    print("Experiment completed.")