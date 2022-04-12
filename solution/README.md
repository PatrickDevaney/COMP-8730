COMP-8730 Project Proposed Solution
===================================

This solution trains a model to identify COVID-19 misinformation in tweets.

Dependencies
------------
The package dependencies are tensorflow, scikit-learn, and nltk.

Running the model also requires the 100-dimension fasttext English word embeddings [available here](https://github.com/pedrada88/crossembeddings-twitter).
By default, the model assumes this is in the same directory, but the path to this on the local system can be changed in embeddingsName in the first line.

To Run
------
To train and run the model:

	python solution.py labels_file training_file predict_file

where labels_file is a file of comma-seperated tweet ID's and labels (1 for misinformation tweets, 0 otherwise), and training_file and predict_file are JSON files of tweets obtained by
Twitter's API. The ID's of all tweets in training_file must appear in labels_file, but any additional labels will be ignored.
