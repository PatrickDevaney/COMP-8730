# Preprocesses EXAMSDAT.643 into a file of misspelled and correct pairs
# Read the file into memory
with open('corpus/EXAMSDAT.643') as f:
	contents = f.read()
# Split into lines
contents = contents.split('\n')
# Break each valid line into pairs
pairs = []
for line in contents:
	line = line.split("  ")
	if len(line) == 2 and 'a' <= line[0][0] <= 'z' and 'a' <= line[1][0] <= 'z':
		pairs.append((line[0],line[1].split(' ')[0]))
with open('preprocessed/EXAMSDAT.643', 'w+') as f:
	for p,q in pairs:
		f.write(p + " " + q + '\n')
