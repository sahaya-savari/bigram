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

# --- Load default corpus at startup ---

base_dir = os.path.dirname(__file__)
corpus_path = os.path.join(base_dir, "corpus.txt")

with open(corpus_path, "r", encoding="utf-8") as f:
    default_corpus_text = f.read()


def build_model(text: str):
    """Build unigram and bigram counts from text."""
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    unigram = defaultdict(int)
    bigram = defaultdict(int)
    for i in range(len(words) - 1):
        unigram[words[i]] += 1
        bigram[(words[i], words[i + 1])] += 1
    return unigram, bigram


def get_prediction(current_word: str, unigram: dict, bigram: dict) -> dict:
    """Predict the next word given unigram and bigram counts."""
    current_word = current_word.lower()
    candidates = {}
    for (w1, w2), count in bigram.items():
        if w1 == current_word:
            candidates[w2] = count / unigram[w1]
    if not candidates:
        return {"input": current_word, "prediction": None, "message": "No prediction available"}
    best = max(candidates, key=candidates.get)
    return {
        "input": current_word,
        "prediction": best,
        "confidence": round(candidates[best], 4),
        "all_candidates": {k: round(v, 4) for k, v in sorted(candidates.items(), key=lambda x: -x[1])},
    }


# Pre-build default model
default_unigram, default_bigram = build_model(default_corpus_text)


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(base_dir, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/predict")
def predict(word: str = Query(..., description="The word to predict the next word for")):
    """Predict using the default corpus."""
    return get_prediction(word, default_unigram, default_bigram)
