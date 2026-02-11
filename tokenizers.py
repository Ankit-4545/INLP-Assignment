import re
from collections import defaultdict, Counter

# Whitespace-based tokenizer
class WhitespaceTokenizer:
    
    def __init__(self):
        self.vocab = set()
    
    def train(self, corpus):
        
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            self.vocab.update(tokens)
    
    def _tokenize(self, text):
        return text.split()
       
    def tokenize(self, text):
        return self._tokenize(text)

class RegexTokenizer:
    
    def __init__(self, pattern=None):
       
        self.pattern = pattern or r"\w+|[^\w\s]"
        self.vocab = set()
    
    def train(self, corpus):

        for sentence in corpus:
            tokens = self.tokenize(sentence)
            self.vocab.update(tokens)
    
    def tokenize(self, text):
        
        tokens = re.findall(self.pattern, text.lower())
        return tokens


class BPETokenizer:
    
    def __init__(self, num_merges=1000):
       
        self.num_merges = num_merges
        self.vocab = set()
        self.merges = []  # List of merge operations
    
    def train(self, corpus):
        
        # Step 1: Initialize with character-level tokens
        word_freqs = self._get_word_frequencies(corpus)
        
        # Step 2: Iteratively merge most frequent pairs
        for i in range(self.num_merges):
            pairs = self._get_pair_frequencies(word_freqs)
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)
        
        # Build final vocabulary
        self._build_vocab(word_freqs)
    
    def _get_word_frequencies(self, corpus):
    
        word_freqs = defaultdict(int)
        
        for sentence in corpus:
            # Split sentence into words (simple whitespace split)
            words = sentence.lower().split()
            for word in words:
                # Split word into characters and add end-of-word marker
                chars = tuple(list(word) + ['</w>'])
                word_freqs[chars] += 1
        
        return word_freqs
    
    def _get_pair_frequencies(self, word_freqs):
    
        pairs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            # Count adjacent pairs in this word
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        
        return pairs
    
    def _merge_pair(self, pair, word_freqs):

        new_word_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                # Check if current position matches the pair
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    # Merge the pair into a single token
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_freqs[tuple(new_word)] += freq
        
        return new_word_freqs
    
    def _build_vocab(self, word_freqs):
        # Extract all unique symbols from word_freqs
        for word in word_freqs.keys():
            for symbol in word:
                self.vocab.add(symbol)
    
    def tokenize(self, text):

        tokens = []
        words = text.lower().split()
        
        for word in words:
            # Start with character-level tokens + end marker
            word_tokens = list(word) + ['</w>']
            
            # Apply learned merges in order
            for merge_pair in self.merges:
                i = 0
                new_tokens = []
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == merge_pair[0] and word_tokens[i + 1] == merge_pair[1]:
                        new_tokens.append(merge_pair[0] + merge_pair[1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            tokens.extend(word_tokens)
        
        return tokens


def clean_corpus(text):

    cleaned = text
    
    # Normalize unicode characters (NFC normalization)
    import unicodedata
    cleaned = unicodedata.normalize('NFC', cleaned)
    
    # Remove zero-width characters and other invisible unicode
    cleaned = re.sub(r'[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]', '', cleaned)
    
    # Replace various unicode dashes/hyphens with standard hyphen
    cleaned = re.sub(r'[\u2010-\u2015\u2212]', '-', cleaned)
    
    # Replace various unicode quotes with standard quotes
    cleaned = re.sub(r'[\u2018\u2019\u201a\u201b]', "'", cleaned)
    cleaned = re.sub(r'[\u201c\u201d\u201e\u201f]', '"', cleaned)
    
    # Replace ellipsis character with three dots
    cleaned = re.sub(r'\u2026', '...', cleaned)
    
    # Remove control characters (except newlines and tabs)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
    
    # Fix excessive whitespace (multiple spaces to single space)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    # Fix excessive newlines (more than 2 consecutive newlines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Remove leading/trailing whitespace from each line
    lines = cleaned.split('\n')
    lines = [line.strip() for line in lines]
    cleaned = '\n'.join(lines)
    
    # Remove empty lines at start and end
    cleaned = cleaned.strip()
    
    return cleaned


def split_corpus(sentences, train_ratio=0.8, val_ratio=0.1):

    n = len(sentences)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = sentences[:train_end]
    val = sentences[train_end:val_end]
    test = sentences[val_end:]
    
    return train, val, test


if __name__ == "__main__":

    test_text = "hello, world"
    
    ws = WhitespaceTokenizer()
    print(f"Whitespace: {ws.tokenize(test_text)}")
    
    regex = RegexTokenizer()
    print(f"Regex: {regex.tokenize(test_text)}")
    
    bpe = BPETokenizer(num_merges=50)
    bpe.train([test_text] * 10)
    print(f"BPE: {bpe.tokenize(test_text)}")
