from heapq import nlargest
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import brown
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from math import inf
import pytrec_eval

# Make an n-gram model from a given tokenized corpus
def makeModel(n, tokenized_text):
	model = MLE(n)
	train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
	model.fit(train_data, padded_sents)
	return model

# Get top ten words predicted by the model
def getTopTen(m, n, pair):
	return [i[1] for i in nlargest(10, [(m.logscore(x, pair[1][-(n-1):]), x) for x in m.vocab])]

# Tokenize a line from the misspelling corpus and return it in a tuple with the next word
def process_line(line):
	ws = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(line)][0]
	return (ws[1], ws[2:])

# Calculate and print s@k
def sAtK(w, s1, s5, s10):
	# Evaluate averages using trac_eval
	qrel = {
	    's@1':w,
	    's@5':w,
	    's@10':w,
	}
	evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map'})
	run = {
	    's@1':s1,
	    's@5':s5,
	    's@10':s10,
	}
	avgs = evaluator.evaluate(run)

	# Extract and print the averages
	s1a = avgs["s@1"]["map"]
	s5a = avgs["s@5"]["map"]
	s10a = avgs["s@10"]["map"]

	print("Average s@1:", s1a)
	print("Average s@5:", s5a)
	print("Average s@10:", s10a)

# Using the Brown news category as our corpus
tokenized_text = [list(map(str.lower, sent)) for sent in brown.sents(categories='news')]

# Read the file into memory
with open('APPLING1DAT.643') as f:
	contents = f.read()
# Split into lines and tokenize (keeping the word-misspelling pairs as the first two words for the moment)
contents = [process_line(x.split("*")[0]) for x in contents.split('\n') if len(x.split("*")) == 2]
ncontents = len(contents)

# Values of n we are using
ns = [1,2,3,5,10]

# Calculate s@k for models with each value of n
for n in ns:
	# The ID of each sentence
	w = {str(i):1 for i in range(ncontents)}
	s1 = {}
	s5 = {}
	s10 = {}
	m = makeModel(n, tokenized_text)
	# Check when the desired word is in the top-n from the model
	for i,pair in enumerate(contents):
		word = pair[0]
		topten = getTopTen(m,n,pair)
		if word == topten[0]:
			s1[str(i)] = 1.0
		if word in topten[:5]:
			s5[str(i)] = 1.0
		if word in topten:
			s10[str(i)] = 1.0
	# Output results
	print("n=" + str(n) + ":")
	sAtK(w,s1,s5,s10)
	print()
