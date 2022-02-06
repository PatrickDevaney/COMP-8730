COMP-8730 Assignment 1
======================

This assignment compares average s@1, s@5, and s@10 for finding corrections to misspelled words using minimum edit distance.

To Run
------
To preprocess the EXAMS file from the Birkbeck corpus into input-output pairs:
	python preprocess.py
This will write EXAMSDAT.643 into the preprocessed directory.

To sample words from a preprocessed file:
	python sample file1 file2 n
where file1 is a preprocessed file with each line of the form
	misspelling correct
in the preprocessed directory, file2 is the name of the file to be written in the samples directory, and n is the number of pairs to sample.
To run an experiment:
	python experiment.py filename
where filename is the name of a file in the samples directory to use as input.

Replication
-----------
The files sample50.txt, sample100.txt, and sample250.txt in samples are the inputs used for the experimental results in my report.
