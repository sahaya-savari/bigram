from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from collections import defaultdict
import re
import os
from typing import Optional

app = FastAPI(
    title="Bigram Language Model ",
    description="A simple next-word prediction using Bigram Language Model",
    version="1.0.0",
)

# --- Corpus and model ---

base_dir = os.path.dirname(__file__)
corpus_path = os.path.join(base_dir, "corpus.txt")

_cached_mtime = None
_cached_unigram = None
_cached_bigram = None


def build_model(text: str):
    """Build unigram and bigram counts from text, respecting sentence boundaries."""
    text = text.lower()
    # Split into sentences so bigrams never cross sentence boundaries
    sentences = re.split(r'[.!?\n]+', text)
    unigram = defaultdict(int)
    bigram = defaultdict(int)
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence)
        for i in range(len(words) - 1):
            unigram[words[i]] += 1
            bigram[(words[i], words[i + 1])] += 1
        if words:
            unigram[words[-1]] += 1  # count last word too
    return unigram, bigram


def get_model():
    """Return the current bigram model, reloading if corpus.txt changed."""
    global _cached_mtime, _cached_unigram, _cached_bigram
    mtime = os.path.getmtime(corpus_path)
    if _cached_mtime is None or mtime != _cached_mtime:
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        _cached_unigram, _cached_bigram = build_model(text)
        _cached_mtime = mtime
    return _cached_unigram, _cached_bigram


def get_next_word(current_word: str, unigram: dict, bigram: dict) -> Optional[str]:
    """Get the most likely next word given bigram counts."""
    current_word = current_word.lower()
    candidates = {}
    for (w1, w2), count in bigram.items():
        if w1 == current_word:
            candidates[w2] = count / unigram[w1]
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def generate_sentence(seed: str, unigram: dict, bigram: dict, max_words: int = 12) -> dict:
    """Generate a sentence by chaining bigram predictions from a seed word."""
    seed_lower = seed.lower()
    words = [seed_lower]
    current = seed_lower
    seen_pairs = set()

    for _ in range(max_words):
        next_word = get_next_word(current, unigram, bigram)
        if next_word is None:
            break
        pair = (current, next_word)
        if pair in seen_pairs:
            break  # avoid loops
        seen_pairs.add(pair)
        words.append(next_word)
        current = next_word

    if len(words) <= 1:
        return {"seed": seed, "sentence": None, "message": "No prediction available for this word"}

    # Capitalise the first word for display
    words[0] = words[0].capitalize()
    sentence = " ".join(words)

    return {"seed": seed, "sentence": sentence, "word_count": len(words)}


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(base_dir, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/predict")
def predict(word: str = Query(..., description="The word to predict the next word for")):
    """Predict the next single word using the default corpus."""
    unigram, bigram = get_model()
    nw = get_next_word(word, unigram, bigram)
    return {"input": word, "prediction": nw}


@app.get("/generate")
def generate(word: str = Query(..., description="Seed word to generate a sentence from")):
    """Generate a full sentence starting from the given seed word."""
    unigram, bigram = get_model()
    return generate_sentence(word, unigram, bigram)
