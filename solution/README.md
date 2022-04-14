COMP-8730 Project Proposed Solution
===================================

This solution trains a model to identify COVID-19 misinformation in tweets.

Dependencies
------------
The package dependencies are tensorflow, scikit-learn, and nltk.

To install the dependencies:

	pip install -r requirements.txt

Running the model also requires the 100-dimension fasttext English word embeddings [available here](https://github.com/pedrada88/crossembeddings-twitter).
By default, the model assumes this is in the same directory, but the path to this on the local system can be changed in embeddingsName in the first line.

Dataset
-------
As a sample dataset, we use the labeled portion of [ANTiVax](https://github.com/SakibShahriar95/ANTiVax), randomly split into training and test sets.

As the Twitter API rules forbid distributing tweets, the datasets are provided as lists of tweet ID's, which will need to be rehydrated. We recommend using [Hydrator](https://github.com/DocNow/hydrator). Instructions for installation are available in their README file. Run the rehydration tool on dataset-train.txt and dataset-test.txt and input the resulting JSON files into our model.

To Evaluate
-----------
To evaluate the model:

	python solution.py -eval labels_file training_file

where labels_file is a file of comma-seperated tweet ID's and labels (1 for misinformation tweets, 0 otherwise), and training_file and predict_file are JSON files of tweets obtained by
Twitter's API. The ID's of all tweets in training_file must appear in labels_file, but any additional labels will be ignored.

The program will automatically split 80% of the data into a training set and 20% into a test set.

To use the provided dataset:

	python solution.py -eval dataset/labels.txt datasets/dataset.jsonl


To Predict
----------
To train the model and make predictions on unlabeled data:

	python solution.py labels_file training_file predict_file

where labels_file is a file of comma-seperated tweet ID's and labels (1 for misinformation tweets, 0 otherwise), and training_file and predict_file are JSON files of tweets obtained by
Twitter's API. The ID's of all tweets in training_file must appear in labels_file, but any additional labels will be ignored.

To use the provided dataset:

	python solution.py dataset/labels-train.txt datasets/dataset-train.jsonl datasets/dataset-test.jsonl
