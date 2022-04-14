# COMP-8730 Assignment 3
# By: Patrick Devaney
from heapq import nlargest
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown
import numpy as np
import pytrec_eval
import scipy
from scipy.spatial.distance import cosine
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

# Download corpus
nltk.download('brown', quiet=True)

# Return top-10 similar words using tf-idf
def tfidf(vocab, corpus):
   # Fit a tf-idf vectorizer on the corpus
   vectorizer = TfidfVectorizer()
   x = vectorizer.fit_transform([x.lower() for x in corpus])
   # Create a set of words in the corpus for O(1) access
   corpus_set = {w for w in corpus}
   tfidf_sim = {}
   for word in vocab:
      if word not in corpus_set:
         continue
      sim = []
      for word_ in vocab:
         if word == word_ or word not in corpus_set:
            continue
         vectors = vectorizer.transform([word,word_])
         sim.append((1.0 - cosine(vectors[0].toarray(),vectors[1].toarray().T), word_))
      largest = nlargest(10, sim)
      tfidf_sim[word] = {x[1]:x[0] for x in largest}
   return tfidf_sim

# Return top-10 similar words using word2vec
def w2v(vocab, corpus, vector_size, window):
   # Tokenize the corpus for word2vec
   tok_corpus = [nltk.word_tokenize(s) for s in corpus]
   wv_model = Word2Vec(sentences=tok_corpus, vector_size=vector_size, window=window, min_count=1, epochs = 1000, workers=8)

   wv_words = []
   wv_similarities = {}

   for word in vocab:
      if word in wv_model.wv:
         wv_words.append(word)

   for word in wv_words:
      sim = []
      for word_ in wv_words:
         if word == word_:
            continue
         sim.append((1 - cosine(wv_model.wv[word], wv_model.wv[word_].T), word_))
      largest = nlargest(10, sim)
      wv_similarities[word] = {x[1]:x[0] for x in largest}
   return wv_similarities

# Returns the average MAP and nDCG when testing terms in t1 against t2
def averageStatistics(t1, t2):
   s_map = 0.0
   s_ndcg = 0.0
   c = 0
   evaluator = pytrec_eval.RelevanceEvaluator(t1, {'map', 'ndcg'})
   result = evaluator.evaluate(t2)
   for r in result:
      s_map += result[r]["map"]
      s_ndcg += result[r]["ndcg"]
      c += 1
   return (s_map / c, s_ndcg / c)

# Read the SimLex-999 file into memory
with open('SimLex-999/SimLex-999.txt') as f:
   contents = f.read()
# Split into lines and get similarity
contents = [y for y in contents.split("\n")[1:-1]]
# Dictionary of similarity scores
simDict = {}
# Create a dictionary of each word's similar words from the read lines
for i in range(len(contents)):
   line = contents[i].split('\t')
   if line[0] not in simDict:
      simDict[line[0]] = {}
   if line[1] not in simDict:
      simDict[line[1]] = {}
   simDict[line[0]][line[1]] = line[3]
   simDict[line[1]][line[0]] = line[3]

# Calculate the (at most) top-10 similar words for each word in SimLex using BFS
top10simlex = {}
for word in simDict:
   scores = simDict[word].copy()
   seen = {word for word in scores}
   while len(scores) < 10:
      newScores = {}
      for w in scores:
         for w_ in simDict[w]:
            if w_ not in seen:
               seen.add(w_)
               newScores[w_] = simDict[w][w_]
      if len(newScores) > 0:
         for w in newScores:
            scores[w] = newScores[w]
         else:
            break
   # Remove the word from its own similarity index
   if word in scores:
      del scores[word]
   largest = nlargest(10, [(scores[w],w) for w in scores])
   top10simlex[word] = {x[1]:(10-y) for y,x in enumerate(largest)}

# Record start time
t = time()
# Test tf-idf on the News and Science Fiction genres from the Brown corpus
for corpus in ["news", "science_fiction"]:
   print("tfidf", corpus)
   print(averageStatistics(top10simlex, tfidf(top10simlex, brown.words(categories=[corpus]))))
# Determine the average time taken by tf-idf
print("tf-idf average time: ", (time() - t) / 2)

# Record start time
t = time()
# Test parameters of word2vec on the News and Science Fiction genres from the Brown corpus
for corpus in ["news", "science_fiction"]:
   for v_size in (10,50,100,300):
      for w_size in (1,2,5,10):
         print("word2vec", corpus, "vector size: ", v_size, ", window size: ", w_size)
         print(averageStatistics(top10simlex, w2v(top10simlex, brown.words(categories=[corpus]), v_size, w_size)))
# Determine average time taken by word2vec
print("word2vec average time: ", (time() - t) / 32)
