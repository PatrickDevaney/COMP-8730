# Select random samples from word pairs in a given text file
from random import shuffle
from sys import argv

if len(argv) < 2:
	exit("Please enter a file name to draw samples from")
if len(argv) < 3:
	exit("Please enter a file name to write to")
if len(argv) < 4:
	exit("Please enter a number of samples to draw")

rfn = argv[1]
wfn = argv[2]

# Read the file to sample into memory
with open('preprocessed/' + rfn) as f:
	contents = f.read()
# Split into lines
contents = contents.split('\n')
# Break each valid line into pairs and make a list
pairs = []
for line in contents:
	line = line.split(" ")
	if len(line) == 2:
		pairs.append((line[0],line[1].split(' ')[0]))
# Shuffle the list
shuffle(pairs)
with open('samples/' + wfn, 'w+') as f:
	c = 0
	n = int(argv[3])
	for p,q in pairs:
		f.write(p + " " + q + '\n')
		c += 1
		if(c >= n):
			break
