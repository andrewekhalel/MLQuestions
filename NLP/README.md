# Natural Language Processing (NLP): Key Concepts Explained

A comprehensive guide to the fundamental concepts in NLP, with clear explanations and practical Python code examples.

---

## 1. What is the difference between stemming and lemmatization?

- **Stemming:** Reduces words to their base or "stem" form, often by chopping off suffixes. Stems may not be valid words.
- **Lemmatization:** Reduces words to their root form ("lemma") using vocabulary and morphological analysis. Lemmas are valid words.

**Key Difference:** Stemming is rule-based and context-free; lemmatization is dictionary-based and context-aware.

<details>
<summary>Code Example (NLTK)</summary>

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
words = ["studies", "studying", "better", "is", "ran"]

print("--- Stemming Example ---")
for word in words:
    print(f"{word} -> {stemmer.stem(word)}")

print("\n--- Lemmatization Example (with POS) ---")
print(f"better (adjective) -> {lemmatizer.lemmatize('better', pos=wordnet.ADJ)}")
print(f"ran (verb) -> {lemmatizer.lemmatize('ran', pos=wordnet.VERB)}")
print(f"is (verb) -> {lemmatizer.lemmatize('is', pos=wordnet.VERB)}")
```
</details>

---

## 2. What do you know about Latent Semantic Indexing (LSI)?

LSI (or LSA) uncovers hidden semantic structures in text. It uses Singular Value Decomposition (SVD) on a term-document matrix (e.g., TF-IDF) to identify underlying "topics" or "concepts".

<details>
<summary>Code Example (scikit-learn)</summary>

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

documents = [
    "The cat sat on the mat.",
    "My dog loves to chase cats.",
    "The new car is a fast automobile.",
    "I saw a car driving on the road.",
    "The cat and the dog are friends."
]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
lsi = TruncatedSVD(n_components=2, random_state=42)
lsi.fit(X)

terms = vectorizer.get_feature_names_out()
print("--- Top terms per topic ---")
for i, comp in enumerate(lsi.components_):
    sorted_terms = sorted(zip(terms, comp), key=lambda x: x[1], reverse=True)[:5]
    print(f"Topic {i+1}: ", end="")
    for t in sorted_terms:
        print(f"{t[0]} ", end="")
    print()
```
</details>

---

## 3. What do you know about Dependency Parsing?

Dependency Parsing analyzes the grammatical structure of a sentence, identifying relationships between "head" words and dependents, forming a dependency tree.

<details>
<summary>Code Example (spaCy)</summary>

```python
import spacy
nlp = spacy.load("en_core_web_sm")
sentence = "The quick brown fox jumps over the lazy dog."
doc = nlp(sentence)

print(f"{'Text':<10} | {'Dependency':<10} | {'Head Text':<10} | {'Head POS':<6}")
print("-" * 50)
for token in doc:
    print(f"{token.text:<10} | {token.dep_:<10} | {token.head.text:<10} | {token.head.pos_:<6}")
```
</details>

---

## 4. Name different approaches for text summarization.

- **Extractive Summarization:** Selects important sentences/phrases from the source.
- **Abstractive Summarization:** Generates new sentences, paraphrasing the source content.

<details>
<summary>Code Example (gensim)</summary>

```python
from gensim.summarization import summarize

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned 
with the interactions between computers and human language, in particular how to program computers to process and 
analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of 
documents, including the contextual nuances of the language within them.
"""

summary = summarize(text, ratio=0.4)
print(summary)
```
</details>

---

## 5. What approach would you use for part of speech tagging?

Modern approaches use statistical (e.g., HMM, CRF) and neural models (RNNs, Transformers). Libraries like NLTK use pre-trained statistical taggers.

<details>
<summary>Code Example (NLTK)</summary>

```python
import nltk
from nltk.tokenize import word_tokenize

sentence = "They are learning about NLP and its various applications."
tokens = word_tokenize(sentence)
tagged_tokens = nltk.pos_tag(tokens)

print(tagged_tokens)
```
</details>

---

## 6. Explain what is a n-gram model.

An n-gram model predicts the next item in a sequence based on the previous (n-1) items. Common: unigrams (1), bigrams (2), trigrams (3).

<details>
<summary>Code Example (NLTK)</summary>

```python
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

sentence = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(sentence.lower())

bigrams = list(ngrams(tokens, 2))
print("--- Bigrams ---")
print(bigrams)

trigrams = list(ngrams(tokens, 3))
print("\n--- Trigrams ---")
print(trigrams)
```
</details>

---

## 7. Explain how TF-IDF measures word importance.

TF-IDF (Term Frequency-Inverse Document Frequency) scores words based on their frequency in a document and rarity across documents. High TF-IDF = important and rare in corpus.

<details>
<summary>Code Example (scikit-learn)</summary>

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

documents = [
    "The sun is shining brightly.",
    "The cat is sleeping in the sun.",
    "The shining cat is very bright."
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print(df_tfidf)
```
</details>

---

## 8. What is perplexity used for?

Perplexity evaluates how well a language model predicts a sequence. Lower perplexity = better predictive power.

<details>
<summary>Code Example (conceptual)</summary>

```python
import numpy as np

def calculate_perplexity(probabilities):
    N = len(probabilities)
    log_probabilities = np.log2(probabilities)
    cross_entropy = - (1.0 / N) * np.sum(log_probabilities)
    perplexity = np.power(2, cross_entropy)
    return perplexity

model_A_probs = [0.8, 0.5, 0.3, 0.9] 
model_B_probs = [0.1, 0.2, 0.1, 0.3] 

print(f"Model A (Decent) Perplexity: {calculate_perplexity(model_A_probs):.2f}")
print(f"Model B (Poor) Perplexity: {calculate_perplexity(model_B_probs):.2f}")
```
</details>

---

## 9. What is Bag-of-Words model?

BoW represents text as the frequency of each word, ignoring grammar and order but keeping track of frequency.

<details>
<summary>Code Example (scikit-learn)</summary>

```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

documents = [
    'The dog is happy.',
    'The cat is sad.',
    'The happy cat plays with the happy dog.'
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
df_bow = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)

print(df_bow)
```
</details>

---

## 10. Explain how the Markov assumption affects the bi-gram model?

The Markov assumption states the probability of a word depends only on its immediate predecessor, simplifying computation but limiting context.

<details>
<summary>Code Example (NLTK)</summary>

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist, ConditionalFreqDist

corpus = "the dog saw the cat. the dog chased the cat. the cat ran away."
tokens = word_tokenize(corpus.lower())
bigrams = list(nltk.bigrams(tokens))
cfd = ConditionalFreqDist(bigrams)

preceding_word = "the"
word_to_predict = "cat"
count_the_cat = cfd[preceding_word][word_to_predict]
count_the = FreqDist(tokens)[preceding_word]
probability = count_the_cat / count_the

print(f"P({word_to_predict} | {preceding_word}) = {probability:.2f}")
```
</details>

---

## 11. What are the most common word embedding methods? Explain each briefly.

- **Count-Based Methods (LSA):** Matrix factorization of word co-occurrence.
- **Prediction-Based Methods (Word2Vec):** Predicts word from context (CBOW) or context from word (Skip-gram).
- **Hybrid Methods (GloVe):** Uses global word co-occurrence statistics.
- **Contextualized Models (BERT):** Embeddings depend on context/sentence.

<details>
<summary>Code Example (gensim Word2Vec)</summary>

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

corpus = [
    "The king is a powerful man.",
    "The queen is a wise woman.",
    "The man and woman are rulers.",
]
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=1)
analogy_result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"'king' - 'man' + 'woman' = {analogy_result}")
```
</details>

---

## 12. What are the first few steps before applying an NLP algorithm to a corpus?

1. **Text Cleaning**
2. **Tokenization**
3. **Stop Word Removal**
4. **Text Normalization (Stemming/Lemmatization)**
5. **Feature Extraction (BoW, TF-IDF, Embeddings)**

<details>
<summary>Code Example (Preprocessing function)</summary>

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)

raw_text = "This is a sample sentence, showing off the PRE-PROCESSING steps!!! 123"
processed_text = preprocess_text(raw_text)
print(f"Processed Text: '{processed_text}'")
```
</details>

---

## 13. List a few types of linguistic ambiguities.

- **Lexical Ambiguity:** One word, multiple meanings (e.g., "bank").
- **Syntactic Ambiguity:** Multiple possible parses (e.g., "I saw the man with the telescope").
- **Semantic Ambiguity:** Meaning of sentence unclear (e.g., "The car hit the pole while it was moving.").
- **Anaphoric Ambiguity:** Unclear pronoun reference.

<details>
<summary>Code Example (WordNet synsets for ambiguity)</summary>

```python
import nltk
from nltk.corpus import wordnet

word = "bank"
synsets = wordnet.synsets(word)

print(f"--- Meanings (Synsets) for '{word}' ---")
for i, syn in enumerate(synsets[:5]): # Show first 5 meanings
    print(f"{i+1}. {syn.name()}: {syn.definition()}")
```
</details>

---

**Happy Learning!**
