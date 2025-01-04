# 13 NLP Interview Questions 2025
A collection of technical interview questions for machine learning and natural language processing positions.

The answer to all of these question were generated using ChatGPT!


### 1. What is the difference between stemming and lemmatization? [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)

Stemming and lemmatization are both techniques used in natural language processing to reduce words to their base form. The main difference between the two is that stemming is a crude heuristic process that chops off the ends of words, while lemmatization is a more sophisticated process that uses vocabulary and morphological analysis to determine the base form of a word. Lemmatization is more accurate but also more computationally expensive.

Example: The word "better"
* Stemming: The stem of the word "better" is likely to be "better" (e.g. by using Porter stemmer)
* Lemmatization: The base form of the word "better" is "good" (e.g. by using WordNetLemmatizer with POS tagger)

### 2. What do you know about Latent Semantic Indexing (LSI)? [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
Latent Semantic Indexing (LSI) is a technique used in NLP and information retrieval to extract the underlying meaning or concepts from a collection of text documents. LSI uses mathematical techniques such as Singular Value Decomposition (SVD) to identify patterns and relationships in the co-occurrence of words within a corpus of text. LSI is based on the idea that words that are used in similar context tend to have similar meanings. 

### 3. What do you know about Dependency Parsing? [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
Dependency parsing is a technique used in natural language processing to analyze the grammatical structure of a sentence, and to identify the relationships between its words. It is used to build a directed graph where words are represented as nodes, and grammatical relationships between words are represented as edges. Each node has one parent and can have multiple children, representing the grammatical relations between the words.

There are different algorithms for dependency parsing, such as the Earley parser, the CYK parser, and the shift-reduce parser. 

### 4. Name different approaches for text summarization. [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
There are several different approaches to text summarization, including:
* Extractive summarization: Selects the most important sentences or phrases from the original text.
* Abstractive summarization: Generates new sentences that capture the key concepts and themes of the original text.
* Latent Semantic Analysis (LSA) based summarization: Uses LSA to identify the key concepts in a text.
* Latent Dirichlet Allocation (LDA) based summarization: Uses LDA to identify the topics in a text.
* Neural-based summarization: Uses deep neural networks to generate a summary.

Each approach has its own strengths and weaknesses and the choice of the approach will depend on the specific use case and the quality of the summary desired.

### 5. What approach would you use for part of speech tagging? [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
There are a few different approaches that can be used for part-of-speech (POS) tagging, such as:
* Rule-based tagging: using pre-defined rules to tag text
* Statistical tagging: using statistical models to tag text
* Hybrid tagging: Combining rule-based and statistical methods
* Neural-based tagging: using deep neural networks to tag text

### 6. Explain what is a n-gram model. [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
An n-gram model is a type of statistical language model used in NLP. It is based on the idea that the probability of a word in a sentence is dependent on the probability of the n-1 preceding words, where n is the number of words in the gram.

The model represents the text as a sequence of n-grams, where each n-gram is a sequence of n words. The model uses the frequency of each n-gram in a large corpus of text to estimate the probability of each word in a sentence, based on the n-1 preceding words.

### 7. Explain how TF-IDF measures word importance. [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document or collection of documents. It is calculated as the product of the term frequency (TF) and the inverse document frequency (IDF) of a word.

The term frequency (TF) of a word is the number of times the word appears in a document, normalized by the total number of words in the document.

The inverse document frequency (IDF) of a word is the logarithm of the total number of documents in the corpus divided by the number of documents in which the word appears.


### 8. What is perplexity used for? [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
Perplexity is a statistical measure used to evaluate the quality of a probability model, particularly language models. It is used to quantify the uncertainty of a model when predicting the next word in a sequence of words. The lower the perplexity, the better the model is at predicting the sequence of words. 

Sure, here's the formula for perplexity in LaTeX format:

Perplexity = $2^{H(D)}$

$H(D) = - {\sum}_{i=1}^{N} {P(w_i)log_2{ P(w_i) }}$ [ref](https://en.wikipedia.org/wiki/Perplexity) 

$w_i$ = the i-th word in the sequence

$N$ = the number of words in the sequence

$P(w_i)$ = the probability of the i-th word according to the model

### 9. What is Bag-of-Worrds model? [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
The bag-of-words model is a representation of text data where a text is represented as a bag (multiset) of its words, disregarding grammar and word order but keeping track of the frequency of each word. It is simple to implement and computationally efficient, but it discards grammatical information and word order, which can be important for some NLP tasks.

### 10. Explain how the Markov assumption affects the bi-gram model? [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
The Markov assumption is an important concept in the bi-gram model, it states that the probability of a word in a sentence depends only on the preceding word. The Markov assumption simplifies the bi-gram model by reducing the number of variables that need to be considered, making the model computationally efficient, but it also limits the context that the model takes into account, which can lead to errors in the probability estimates. In practice, increasing the order of the n-gram model can be used to increase the context taken into account, thus increasing the model's accuracy.

### 11. What are the most common word embedding methods? explain each briefly. [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
Common word embedding methods include:
* Count-based methods: Create embeddings by counting the co-occurrence of words in a corpus. Example: Latent Semantic Analysis (LSA)
* Prediction-based methods: Create embeddings by training a model to predict a target word based on its surrounding context. Example: Continuous Bag-of-Words (CBOW) and Word2Vec
* Hybrid methods: Combine both co-occurrence and context to generate embeddings. Example: GloVe (Global Vectors for Word Representation)
* Neural Language Model based methods: Create embeddings by training a neural network-based language model on a large corpus of text. Example: BERT (Bidirectional Encoder Representations from Transformers)

### 12. What are the first few steps that you will take before applying an NLP algorithm to a given corpus? [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
* Text pre-processing: Clean and transform the text into a format that can be processed by the model. Specific methods include: Removing special characters, lowercasing, removing stop words.

* Tokenization: Break the text into individual words or phrases that can be used as input. Specific methods include: word tokenization, sentence tokenization, and n-gram tokenization.

* Text normalization: Transform the text into a consistent format. Specific methods include: stemming, lemmatization.

* Feature extraction: Select relevant features from the text to be used as input. Specific methods include: creating a vocabulary of the most common words in the corpus, creating a term-document matrix.

* Splitting the data: Divide the data into training, validation and testing sets.

* Annotating the data: Manually tag the data with relevant information. Specific methods include: POS tagging, NER tagging, and so on.

### 13. List a few types of linguistic ambiguities. [[src]](https://www.projectpro.io/article/nlp-interview-questions-and-answers/439)
* Lexical ambiguity: A word has multiple meanings. Example: "bass" can refer to a type of fish or a low-frequency sound.

* Syntactic ambiguity: A sentence can be parsed in more than one way. Example: "I saw the man with the telescope" can mean that the speaker saw a man who had a telescope or the speaker saw a man through a telescope.

* Semantic ambiguity: A word or phrase can have more than one meaning in a given context. Example: "bank" can refer to a financial institution or the edge of a river.

* Pragmatic ambiguity: A sentence can have different interpretations depending on the speaker's intended meaning. Example: "I'm fine" can mean that the speaker is feeling well or that the speaker does not want to talk about their feelings.

* Anaphora resolution: A pronoun or noun phrase refers to an antecedent with multiple possible referents.

* Homonymy: Words that are written and pronounced the same but have different meanings. Example: "bass" as a type of fish and a low-frequency sound

* Polysemy: words that have multiple meanings but are related in some way. Example: "bass" as a low-frequency sound and the bass guitar.
