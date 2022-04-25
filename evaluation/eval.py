embeddingsName = "fasttext_english_twitter_100d.vec"

from datetime import datetime
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import random
import re
import pandas as pd
from string import punctuation
from sys import argv
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow.keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, Flatten, GRU, Reshape, TextVectorization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set random seeds to a fixed value
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# Load tweets from a JSON file
def tweetsFromJSON(tweetName):
	# Read the tweets into memory
	with open(tweetName, encoding="utf8") as f:
		contents = f.read()
	# Split into lines and return
	return [json.loads(line) for line in contents.split('\n')[:-1]]

# Read labels from a file
def readLabels(labelName):
	# Read the labels into memory
	with open(labelName) as f:
		contents = f.read()
	# Split into lines
	contents = contents.split('\n')
	# Break each valid line into pairs
	labels = {}
	for line in contents:
		line = line.split(",")
		if len(line) == 2:
			labels[line[0]] = float(line[1])
	return labels

# Read the labellings, tweets, and embeddings for training into memory from file paths
def readLearningFiles(labelName, tweetName, embeddingsName):
	labels = readLabels(labelName)

	# Read tweets into memory
	tweets = tweetsFromJSON(tweetName)

	# Read the vector embeddings into memory
	with open(embeddingsName, encoding="utf8") as f:
		contents = f.read()
	# Split into lines
	contents = contents.split('\n')[1:-1]
	# Store the index of each word in the matrix
	word_vector = {}
	# Split the line, and index each word embedding
	for line in contents:
		line = line.split(" ")
		word_vector[line[0]] = np.array([float(x) for x in line[1:-1]])
	
	return labels, tweets, word_vector

# Count relevant features other than the embedded text in a tweet
def countFeatures(tweet):
	# Counts
	tokens = len(nltk.word_tokenize(tweet["full_text"]))
	emoji = 0
	hashtags = len(tweet["entities"]["hashtags"])
	mentions = len(tweet["entities"]["user_mentions"])
	uppercase = 0
	punct = 0
	exclamation = 0
	urls = len(tweet["entities"]["urls"])
	symbols = len(tweet["entities"]["symbols"])
	# Calculate the difference between the tweet date and account creation date (normalized)
	cr = datetime.strptime(tweet["created_at"], '%a %b %d %H:%M:%S +0000 %Y')
	ac_cr = datetime.strptime(tweet["user"]["created_at"], '%a %b %d %H:%M:%S +0000 %Y')
	acct_age = abs(cr - ac_cr).days / 365.0
	# Timestamp of tweet feature (normalized)
	timestamp = (cr - datetime(2006,1,1)).days / (365.0*16.0)
	# Increment counts from tweet text manually
	for c in tweet["full_text"]:
		if 0x00A0 <= ord(c) <= 0x1FAFF:
			emoji += 1
		if "A" <= c <= "Z":
			uppercase += 1
		if c in punctuation:
			punct += 1
		if c == "!":
			exclamation += 1
	return [tokens, emoji, hashtags, mentions, uppercase, punct, exclamation, urls, symbols, acct_age, timestamp]

# Create vectorizer, embedding matrix, tf-idf vectorizer from a set of tweets and a dict of word embeddings
def preprocessEmbeddings(tweets, word_vector):
	# Create a collection of all training tweet text and train a tokenizer on it
	tweet_text = [x["full_text"] for x in tweets]
	vec = TextVectorization(output_sequence_length = 140)
	tweet_dataset = tf.data.Dataset.from_tensor_slices(tweet_text).batch(32)
	vec.adapt(tweet_dataset)

	# Store the vector embeddings of each word in the embedding matrix
	embedding_matrix = np.zeros((len(vec.get_vocabulary()), 100))
	for i, word in enumerate(vec.get_vocabulary()):
		if word in word_vector:
			embedding_matrix[i] = word_vector[word]

	# Calculate the tf-idf, considering the top 500 words
	t_vec = TfidfVectorizer(max_features=500)
	t_vec.fit(tweet_text)

	return vec, embedding_matrix, t_vec

# Get (optionally labelled) data from tweets formatted into the proper format for neural network input
def preprocessData(vec, t_vec, tweets, labels=None):
	tr_x = []
	tr_y = []
	tr_count_x = []
	tweets = tweets[:]
	np.random.shuffle(tweets)
	for t in tweets:
		tr_x.append(t["full_text"])
		tr_count_x.append(countFeatures(t))
		if labels != None:
			tr_y.append(labels[t["id_str"]])

	tfidf_x = t_vec.transform(tr_x).toarray()
	tr_x = vec(np.array([[i] for i in tr_x])).numpy()
	tr_y = np.array(tr_y)
	tr_count_x = np.array(tr_count_x)
	if labels != None:
		return tr_x, tfidf_x, tr_count_x, tr_y
	return tr_x, tfidf_x, tr_count_x

# Split tweets into test and training sets
def splitData(tweets):
	tweets = tweets[:]
	np.random.shuffle(tweets)
	sp = round(0.8 * len(tweets))
	return tweets[:sp], tweets[sp:]

# Split the tweets for ten-fold cross-validation
def tenFold(tweets):
	tweets = tweets[:]
	np.random.shuffle(tweets)
	sp = round(0.1 * len(tweets))
	return [(tweets[i*sp:(i+1)*sp], np.concatenate((tweets[:i*sp], tweets[(i+1)*sp:]))) for i in range(10)]

# Count true/false positive/negatives
def correctCounts(model, xs, y):
	predictions = model.predict(xs)
	tp = tn = fp = fn = 0
	for i in range(len(predictions)):
		if predictions[i][0] >= 0.5:
			if y[i] >= 0.5:
				tp += 1
			else:
				fp += 1
		else:
			if y[i] >= 0.5:
				fn += 1
			else:
				tn += 1
	return (tp, tn, fp, fn)

# Calculate precision, recall, and f1 scores from true/false positive/negatives
def scores(tp,tn,fp,fn):
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	f1 = (2*precision*recall)/(precision+recall)
	return (precision,recall,f1)

# Create and train neural network
def createRNN(inputs, labels, embedding_matrix=None, batch_size=32, epochs=10, l_2 = 0.01):
	# Get input from word embeddings and feed it to recurrent layers
	input_vec = Input(shape=(None,), dtype="int64")
	embedded = Embedding(len(vec.get_vocabulary()), 100,
		embeddings_initializer=Constant(embedding_matrix), trainable=False)(input_vec)
	r_l = GRU(100, return_sequences=True)(embedded)
	r_l = GRU(100, return_sequences=True)(r_l)
	r_l = GRU(100)(r_l)
	r_l = Flatten()(r_l)
	# Take other features and concatenate them
	tfidf = Input(shape=(500,))
	count = Input(shape=(11,))
	con = Concatenate()([tfidf, count,r_l])
	# Feed all of this to a few regular neural network layers
	d_l = Dense(300, activation="elu", kernel_initializer="he_normal", kernel_regularizer=l2(l_2))(con)
	d_l = Dense(100, activation="elu", kernel_initializer="he_normal", kernel_regularizer=l2(l_2))(d_l)
	output = Dense(1, activation="sigmoid")(d_l)
	# Create, train, and return the final neural network
	rnn = Model([input_vec, tfidf, count],output)
	rnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	rnn.fit(inputs, labels, batch_size=batch_size, epochs=epochs)
	return rnn

# Evaluate the neural network
def evalRNN(vec, t_vec, training, test, labels, l_2 = 0.01):
	# Preprocess training data
	tr_x, tfidf_x, tr_count_x, tr_y = preprocessData(vec, t_vec, training, labels)
	# Preprocess test data
	test_x, test_tfidf_x, test_count_x, test_y = preprocessData(vec, t_vec, test, labels)
	# Create and train the neural network
	rnn = createRNN([tr_x, tfidf_x, tr_count_x], tr_y, embedding_matrix, epochs=25, l_2 = l_2)
	loss, acc = rnn.evaluate([test_x, test_tfidf_x, test_count_x], test_y)  
	return (loss, acc, correctCounts(rnn, [test_x, test_tfidf_x, test_count_x], test_y))

# Create and train BERT neural network
def createBERT(inputs, labels, batch_size=8, epochs=10, dropout=0.1):
	# Get input from word embeddings and feed it to recurrent layers
	input_vec = Input(shape=(), dtype=tf.string)
	preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
	encoder_inputs = preprocessor(input_vec)
	encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4", trainable=True)
	enc_outputs = encoder(encoder_inputs)
	# Feed all of this to a few regular neural network layers
	d_l = enc_outputs['pooled_output']
	d_l = Dropout(dropout)(d_l)
	output = Dense(1, activation="sigmoid")(d_l)
	# Create, train, and return the final neural network
	dnn = Model(input_vec,output)
	dnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	dnn.fit(inputs, labels, batch_size=batch_size, epochs=epochs)
	return dnn

# Get (optionally labelled) data from tweets formatted into the proper format for BERT neural network input
def preprocessBERT(tweets, labels=None):
	tr_x = []
	tr_y = []
	tweets = tweets[:]
	np.random.shuffle(tweets)
	for t in tweets:
		tw = t["full_text"].lower()
		# Remove text within parentheses
		tw = re.sub(r'\([^)]*\)', '', tw)
		# Tokenize the words of the tweet
		tw = nltk.word_tokenize(tw)
		# The processed list of words
		tw_p = []
		# Create a stemmer
		stemmer = SnowballStemmer(language="english")
		for w in tw:
			# Stem the word
			w = stemmer.stem(w)
			# Eliminate non-word text and stopwords
			if not w.isalpha() or w in stopwords.words("english"):
				continue
			tw_p.append(w)
		# Concatenate the final result
		tw = ""
		for w in tw_p:
			tw += w + " "
		tw = tw[:-1]
		tr_x.append(tw)
		if labels != None:
			tr_y.append(labels[t["id_str"]])
	tr_x = np.array(tr_x)
	tr_y = np.array(tr_y)
	if labels != None:
		return tr_x, tr_y
	return tr_x

# Evaluate the BERT model
def evalBERT(training, test, labels, dropout=0.1):
	# Preprocess training data
	tr_x, tr_y = preprocessBERT(training, labels)
	# Preprocess test data
	test_x, test_y = preprocessBERT(test, labels)
	# Create and train the neural network
	dnn = createBERT(tr_x, tr_y, epochs=3, dropout=dropout)
	loss, acc = dnn.evaluate(test_x, test_y)  
	return (loss, acc, correctCounts(dnn, test_x, test_y))

# Require all file name arguments
if len(argv) < 5:
	print("Error: Provide method, labels and training data")
	exit()

# Get method and parameter
method = argv[1]
param = float(argv[2])

# Get label and dataset filenames
labelName = argv[3]
tweetName = argv[4]

# Load the files into memory
labels, tweets, word_vector = readLearningFiles(labelName, tweetName, embeddingsName)

# Preprocess embeddings
vec, embedding_matrix, t_vec = preprocessEmbeddings(tweets,word_vector)


# Split tweets into 10-fold cross-validation sets
datasets = tenFold(tweets)
avgLoss = avgAcc = 0.0
tp = tn = fp = fn = 0
for test, training in datasets:
	if argv[1] == "-rnn":
		loss, acc, counts = evalRNN(vec, t_vec, training, test, labels, param)
	elif argv[1] == "-bert":
		loss, acc, counts = evalBERT(training, test, labels, param)
	avgLoss += loss
	avgAcc += acc
	tp += counts[0]
	tn += counts[1]
	fp += counts[2]
	fn += counts[3]
print("Average Loss:", (avgLoss/10), "\tAverage Accuracy: ", (avgAcc/10))
p,r,f = scores(tp,tn,fp,fn)
print("Precision: ", p, "\tRecall: ", r, "\tf1: ", f)
