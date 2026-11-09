# Language Technology labs (729G86, Linköping University)

My solutions to the lab assignments in the [Language Technology](https://liu-nlp.ai/lang-tech/) course at Linköping University. The labs move from classic NLP pipelines (tokenization, Naive Bayes, n-gram models) to transformer models and instruction-tuned LLMs.

| Lab | Topic | Notebook |
|-----|-------|----------|
| L0 | Linguistic analysis with spaCy | [intro_to_spacy.ipynb](labs/l0/intro_to_spacy.ipynb) |
| L1 | Text classification | [text_classification.ipynb](labs/l1/text_classification.ipynb) |
| L2 | N-gram language models | [ngram_language_models.ipynb](labs/l2/ngram_language_models.ipynb) |
| L3 | Named entity recognition | [named_entity_recognition.ipynb](labs/l3/named_entity_recognition.ipynb) |
| L4 | Large language models | [large_language_models.ipynb](labs/l4/large_language_models.ipynb) |

## What each lab covers

### L0: Linguistic analysis with spaCy
- Compared spaCy's tokenizer with naive whitespace splitting, which leaves punctuation attached to words (`author,` and `columnist.`).
- Extracted lemmas, part-of-speech tags and morphological features, and filtered tokens by POS, for example to pull out all the verbs in a text.
- Ran spaCy's named entity recognizer and visualized the entities.

### L1: Classifying Swedish parliamentary speeches
Predicting whether a speaker in the Riksdag belongs to the left-wing or right-wing bloc, using only the text of their speech (data from [Riksdagens öppna data](https://data.riksdagen.se/)).
- Built a bag-of-words + Multinomial Naive Bayes pipeline with scikit-learn and inspected the sparse `CountVectorizer` features.
- Implemented accuracy, precision, recall and F1 from scratch and checked them against scikit-learn.
- Compared against a most-frequent-class baseline:

| Model | Accuracy | F1 (left) | F1 (right) |
|-------|---------:|----------:|-----------:|
| Most-frequent-class baseline | 0.61 | 0.00 | 0.76 |
| Naive Bayes (bag of words) | **0.83** | **0.78** | **0.86** |

### L2: N-gram language models
Trained n-gram models (n = 1–4) on the Sherlock Holmes stories.
- Generated text from each model and analysed how fluency improves with order (from random word salad at n = 1 to fluent sentences at n = 4), along with the risk of overfitting to the training text.
- Measured perplexity on the training data, where it drops from 464 (unigram) to 2.5 (4-gram).
- Explored the model's internals by looking up n-gram counts, vocabulary size and continuations.
- Evaluated on held-out text (*The Case-Book of Sherlock Holmes*). Without smoothing, any unseen n-gram makes perplexity infinite. With add-k smoothing, the bigram model with k = 0.01 did best (perplexity ≈ 208), which shows the bias–variance trade-off between too little and too much smoothing.

### L3: Named entity recognition with BERT
- Ran a pretrained BERT NER model (`dslim/bert-base-NER`) through the Hugging Face `pipeline` on 1,000 sentences from the English PUD treebank.
- Did a qualitative error analysis and implemented span-level precision and recall.
- Per-label results:

| Label | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| PER | 0.90 | 0.91 | 0.91 |
| LOC | 0.73 | 0.88 | 0.80 |
| ORG | 0.69 | 0.51 | 0.58 |
| MISC | 0.00 | 0.00 | 0.00 |

  MISC scores 0 because the gold annotations have no MISC label, so every MISC prediction counts as a false positive. That pulls the macro average down even though PER and LOC are strong.

### L4: Working with LLMs (Gemma 3)
- Built a small English–Spanish parallel dataset and computed **tokenizer fertility** (tokens per word) to compare how efficiently the Gemma tokenizer handles each language.
- Compared the base model (`gemma-3-4b-pt`), which only continues text, with the instruction-tuned model (`gemma-3-4b-it`), which follows chat instructions.
- Wrote a `translate()` function using the chat pipeline and evaluated the translation quality by hand.
- Designed a prompt for multiple-choice question answering on **ARC-Easy** and wrote an accuracy function that parses the answer label out of the model's output. The model reached **0.90 accuracy**, against a 25% random baseline.

## Running the notebooks

The environment is managed with [`uv`](https://docs.astral.sh/uv/) and needs Python 3.13 or newer.

```bash
uv run jupyter lab
```

The first run installs all dependencies into a local virtual environment. L3 and L4 download models from Hugging Face. L4 uses 4B-parameter Gemma models, which need a GPU and accepting Gemma's licence on Hugging Face.

## Credits

The course was designed by Marcel Bollmann and colleagues at Linköping University. The lab instructions, the helper package in [`src/liu_lt/`](src/liu_lt/) and the datasets were provided by the course. The task solutions and analysis in the notebooks are my own.
