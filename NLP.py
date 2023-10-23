"""
Setup

Install necessary libraries & download models here

pip install spacy
python -m spacy download en_core_web_md
pip install scikit-learn

Bag of Words

Define some training utterances
"""

class Category:
  BOOKS = "BOOKS"
  CLOTHING = "CLOTHING"

train_x = ["i love the book", "this is a great book", "the fit is great", "i love the shoes"]
train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]

"""
Fit vectorizer to transform text to bag-of-words vectors
"""

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)
train_x_vectors = vectorizer.fit_transform(train_x)

print(vectorizer.get_feature_names_out())
print(train_x_vectors.toarray())

"""
Train SVM (Support Vector Machine finding hyperplanes between points) Model
"""

from sklearn.svm import SVC

clf_svm = SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

"""
Test new utterances on trained model
"""

test_x = vectorizer.transform(['i love the books'])

clf_svm.predict(test_x)

"""
Word Vectors
Skip-Gram: Mapping syntactic and semantic meaning into a latent space by encoding relationships between words by predicting words around words
"""

import spacy

nlp = spacy.load("en_core_web_md")

print(train_x)

"""
Avaraging the word embeddeding for each of the words in the string
"""

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

print(docs)

from sklearn import svm

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)

test_x = ["I went to the bank and wrote a check", "let me check that out"]
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors =  [x.vector for x in test_docs]

clf_svm_wv.predict(test_x_word_vectors)

"""
Regexes
"""

import re

regexp = re.compile(r"\bread\b|\bstory\b|book")

phrases = ["I liked that story.", "the car treaded up the hill", "this hat is nice"]

matches = []
for phrase in phrases:
  if re.search(regexp, phrase):
    matches.append(phrase)

print(matches)

"""
Stemming/Lemmatization

Lemmatization considers the context and converts the word to its meaningful base form, which is called Lemma.
Stemming is a process that stems or removes last few characters from a word, often leading to incorrect meanings and spelling.

Setup
"""

import nltk

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

"""
Stemming
"""

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

phrase = "reading the books"
words = word_tokenize(phrase)

stemmed_words = []
for word in words:
  stemmed_words.append(stemmer.stem(word))

" ".join(stemmed_words)

"""
Lemmatizing
"""

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

phrase = "reading the books"
words = word_tokenize(phrase)

lemmatized_words = []
for word in words:
  lemmatized_words.append(lemmatizer.lemmatize(word, pos='v'))

" ".join(lemmatized_words)

"""
Stopwords
Tokenize, then remove Stopwords
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

phrase = "Here is an example sentence demonstrating the removal of stopwords"

words = word_tokenize(phrase)

stripped_phrase = []
for word in words:
  if word not in stop_words:
    stripped_phrase.append(word)

" ".join(stripped_phrase)

"""
Various other techniques (spell correction, sentiment, & pos tagging)

python -m textblob.download_corpora
"""

from textblob import TextBlob

phrase = "the book was horrible"

tb_phrase = TextBlob(phrase)

tb_phrase.correct()

tb_phrase.tags

tb_phrase.sentiment

"""
Transformer Architecture

The attention mechanism in the transformer architecture allows the model to focus on different parts of the input sequence when producing an output, similar to how humans pay attention to specific parts of a sentence to derive meaning

Setup

pip install spacy-transformers
python -m spacy download en_trf_bertbaseuncased_lg

Using Spacy to utilize BERT Model
"""

import spacy
import torch

nlp = spacy.load("en_trf_bertbaseuncased_lg")
doc = nlp("Here is some text to encode.")

class Category:
  BOOKS = "BOOKS"
  BANK = "BANK"

train_x = ["good characters and plot progression", "check out the book", "good story. would recommend", "novel recommendation", "need to make a deposit to the bank", "balance inquiry savings", "save money"]
train_y = [Category.BOOKS, Category.BOOKS, Category.BOOKS, Category.BOOKS, Category.BANK, Category.BANK, Category.BANK]

from sklearn import svm

docs = [nlp(text) for text in train_x]
train_x_vectors = [doc.vector for doc in docs]
clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors, train_y)

test_x = ["check this story out"]
docs = [nlp(text) for text in test_x]
test_x_vectors = [doc.vector for doc in docs]

clf_svm.predict(test_x_vectors)
