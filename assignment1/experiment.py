from heapq import nsmallest
import json
from levenshtein import levenshtein
from multiprocessing import Pool
import pytrec_eval
from sys import argv

# Get all words from NLTK
from nltk.corpus import wordnet as wn
words = [n for n in wn.all_lemma_names()]

# Get the n (default 10) words within minimum distance of w in sorted order
def minDistances(w,n=10):
	l = []
	for dictw in words:
		l.append((levenshtein(w, dictw), dictw))
	rList = []
	for weight,word in nsmallest(n,l):
		rList.append(word)
	return (w, rList)

# Read the file into memory
with open('samples/' + argv[1]) as f:
	contents = f.read()
# Split into lines
contents = contents.split('\n')[:-1]
# Misspelled words
ws = []
# Dictionary of their correct spellings
correct = {}
# Break each valid line into pairs
for line in contents:
	line = line.split(" ")
	ws.append(line[0])
	correct[line[0]] = line[1]

# Run the results in 8 threads and aggregate the results
with Pool(8) as p:
	results = p.map(minDistances, ws)
results = [x for x in results]

# Dictionary of the misspelled and correct words as the ground truth
w1 = {w+correct[w]:1 for w in ws}
# Dictionaries for positive s@1, s@5, and s@10 results
s1 = {}
s5 = {}
s10 = {}

for w,r in results:
	cw = correct[w]
	if cw == r[0]:
		s1[w+cw] = 1.0
	if cw in r[:5]:
		s5[w+cw] = 1.0
	if cw in r:
		s10[w+cw] = 1.0

# Evaluate averages using trac_eval
qrel = {
    's@1':w1,
    's@5':w1,
    's@10':w1,
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
