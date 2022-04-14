embeddingsName = "fasttext_english_twitter_100d.vec"

from datetime import datetime
import json
import numpy as np
import nltk
import random
from string import punctuation
from sys import argv
import tensorflow as tf

import tensorflow.keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Flatten, GRU, Reshape, TextVectorization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

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

# Create and train neural network
def createRNN(inputs, labels, embedding_matrix=None, batch_size=32, epochs=10):
	# Get input from word embeddings and feed it to recurrent layers
	input_vec = Input(shape=(None,), dtype="int64")
	embedded = Embedding(len(vec.get_vocabulary()), 100,
		embeddings_initializer=Constant(embedding_matrix), trainable=False)(input_vec)
	r_l = GRU(100, return_sequences=True)(embedded)
	r_l = GRU(100, return_sequences=True)(r_l)
	r_l = GRU(100)(r_l)
	# Take other features and concatenate them
	tfidf = Input(shape=(500,))
	count = Input(shape=(11,))
	con = Concatenate()([tfidf, count,r_l])
	# Feed all of this to a few regular neural network layers
	d_l = Dense(300, activation="elu", kernel_initializer="he_normal", kernel_regularizer=l2(0.01))(con)
	d_l = Dense(100, activation="elu", kernel_initializer="he_normal", kernel_regularizer=l2(0.01))(d_l)
	output = Dense(1, activation="sigmoid")(d_l)
	# Create, train, and return the final neural network
	rnn = Model([input_vec, tfidf, count],output)
	rnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	rnn.fit(inputs, labels, batch_size=batch_size, epochs=epochs)
	return rnn

# Require all file name arguments
if len(argv) < 4:
	print("Error: Provide labels, training data, and test data")
	exit()

# Determine names based on presence of an eval flag
if argv[1] == "-eval":
	labelName = argv[2]
	tweetName = argv[3]
else:
	labelName = argv[1]
	tweetName = argv[2]
	predictName = argv[3]

# Load the files into memory
labels, tweets, word_vector = readLearningFiles(labelName, tweetName, embeddingsName)

# Preprocess embeddings
vec, embedding_matrix, t_vec = preprocessEmbeddings(tweets,word_vector)

if argv[1] == "-eval":
	# Split tweets into training and test
	trTweets, testTweets = splitData(tweets)
	# Preprocess training data
	tr_x, tfidf_x, tr_count_x, tr_y = preprocessData(vec, t_vec, trTweets, labels)
	# Preprocess test data
	test_x, test_tfidf_x, test_count_x, test_y = preprocessData(vec, t_vec, trTweets, labels)
	# Create and train the neural network
	rnn = createRNN([tr_x, tfidf_x, tr_count_x], tr_y, embedding_matrix)
	loss, acc = rnn.evaluate([test_x, test_tfidf_x, test_count_x], test_y)
	print("Loss: ", loss, "\tAccuracy: ", acc)
else:
	# Preprocess training data
	tr_x, tfidf_x, tr_count_x, tr_y = preprocessData(vec, t_vec, tweets, labels)
	# Create and train the neural network
	rnn = createRNN([tr_x, tfidf_x, tr_count_x], tr_y, embedding_matrix)
	# Load data to predict and make preductions
	predictTweets = tweetsFromJSON(predictName)
	predict_tr, predict_tfidf, predict_count = preprocessData(vec, t_vec, predictTweets)
	predictions = rnn.predict([predict_tr, predict_tfidf, predict_count])
	for i in range(len(predictTweets)):
		print(predictTweets[i]["id_str"] + "," + str(predictions[i][0]))
