# Bigram Language Model — Next Word Predictor

A Bigram Language Model that predicts the next word given an input word, using conditional probability from a text corpus.

---

## Step-by-Step Code Explanation

### Step 1: Import Libraries

```python
from collections import defaultdict
import re
```

- **`defaultdict`** — A special dictionary that automatically initializes missing keys with a default value (here, `int` gives `0`). This avoids `KeyError` when counting words.
- **`re`** — Regular expression module used to split text into clean words.

---

### Step 2: Read the Corpus

```python
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()
```

- Opens `corpus.txt` and reads the entire content into a string.
- `.lower()` converts everything to lowercase so that `"The"` and `"the"` are treated as the same word.
- `encoding="utf-8"` ensures special characters are handled properly.

---

### Step 3: Tokenize the Text

```python
words = re.findall(r'\b\w+\b', text)
```

- `re.findall()` extracts all words from the text using a regex pattern.
- `\b\w+\b` means: find sequences of word characters (letters, digits, underscore) bounded by word boundaries.
- **Result:** A list of clean words like `["the", "tiger", "is", "a", "powerful", ...]`
- Punctuation (periods, commas) is automatically excluded.

---

### Step 4: Count Unigrams and Bigrams

```python
unigram_count = defaultdict(int)
bigram_count = defaultdict(int)

for i in range(len(words) - 1):
    unigram_count[words[i]] += 1
    bigram_count[(words[i], words[i + 1])] += 1
```

- **Unigram** = a single word. `unigram_count` stores how many times each word appears.
  - Example: `{"the": 6, "tiger": 5, "is": 3, ...}`
- **Bigram** = a pair of consecutive words. `bigram_count` stores how many times each pair appears.
  - Example: `{("the", "tiger"): 5, ("tiger", "is"): 2, ...}`
- The loop goes through every word (except the last) and counts both.
- `len(words) - 1` is used because the last word has no next word to form a bigram.

---

### Step 5: Prediction Function

```python
def predict_next_word(current_word):
    current_word = current_word.lower()
    candidates = {}

    for (w1, w2), count in bigram_count.items():
        if w1 == current_word:
            candidates[w2] = count / unigram_count[w1]

    if not candidates:
        return "No prediction available"

    return max(candidates, key=candidates.get)
```

**How it works:**

1. Converts the input word to lowercase for matching.
2. Loops through all bigrams and finds those that start with the input word.
3. For each matching bigram `(w1, w2)`, calculates the **conditional probability**:

   ```
   P(w2 | w1) = Count(w1, w2) / Count(w1)
   ```

   - Example: If `("the", "tiger")` appears 5 times and `"the"` appears 6 times → P = 5/6 = 0.833

4. Stores all candidate next words with their probabilities in `candidates`.
5. If no bigrams start with the input word → returns `"No prediction available"`.
6. Otherwise, returns the word with the **highest probability** using `max()`.

---

### Step 6: User Input and Output

```python
user_word = input("Enter a word: ")
next_word = predict_next_word(user_word)
print(f"Next word prediction: {next_word}")
```

- Asks the user to type a word.
- Calls the prediction function with that word.
- Prints the predicted next word.

---

## Example Run

```
Enter a word: the
Next word prediction: tiger
```

This means in the corpus, the word **"tiger"** most frequently follows **"the"**.

---

## Key Concepts

| Term | Meaning |
|------|---------|
| **Unigram** | A single word and its count |
| **Bigram** | A pair of two consecutive words |
| **Conditional Probability** | The chance of word B appearing after word A |
| **`defaultdict(int)`** | Dictionary that defaults missing keys to `0` |
| **`re.findall()`** | Extracts all pattern matches from text |

---

## Corpus File (`corpus.txt`)

The corpus is a small text about tigers. You can replace it with any text — the more text you provide, the better the predictions.

Current corpus:
```
The tiger is a powerful animal. The tiger lives in forests and grasslands.
The tiger hunts deer and wild boar. A tiger is known for its strength and stripes.
The tiger moves silently while hunting. The tiger is a symbol of courage.
```
