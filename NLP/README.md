# Natural Language Processing Interview Questions
A collection of technical interview questions for machine learning and computer vision engineering positions.

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



