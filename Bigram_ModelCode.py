from collections import defaultdict
import re

# ---------- STEP 1: Read corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# ---------- STEP 2: Tokenize ----------
words = re.findall(r'\b\w+\b', text)

# ---------- STEP 3: Count unigrams and bigrams ----------
unigram_count = defaultdict(int)
bigram_count = defaultdict(int)

for i in range(len(words) - 1):
    unigram_count[words[i]] += 1
    bigram_count[(words[i], words[i + 1])] += 1

# ---------- STEP 4: Prediction function ----------
def predict_next_word(current_word):
    current_word = current_word.lower()
    candidates = {}

    for (w1, w2), count in bigram_count.items():
        if w1 == current_word:
            candidates[w2] = count / unigram_count[w1]  # Conditional Probability

    if not candidates:
        return "No prediction available"

    return max(candidates, key=candidates.get)

# ---------- STEP 5: USER INPUT ----------
user_word = input("Enter a word: ")
next_word = predict_next_word(user_word)

print(f"Next word prediction: {next_word}")