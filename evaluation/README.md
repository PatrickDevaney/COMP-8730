COMP-8730 Project Evaluation
============================

This evaluation uses 10-fold cross-validation on our proposed model to identify COVID-19 misinformation in tweets.

Dependencies
------------
To install the dependencies:

	pip install -r requirements.txt

Running the model also requires the 100-dimension fasttext English word embeddings [available here](https://github.com/pedrada88/crossembeddings-twitter). Specifically, we use fasttext_english_twitter_100d.vec, which can be downloaded directly from [here](https://drive.google.com/drive/folders/1a9llDhoM6zD-sOKiM0AdSxDYq2-15PJD?usp=sharing).
By default, the model assumes this is in the same directory, but the path to this on the local system can be changed in embeddingsName in the first line.

Dataset
-------
As a sample dataset, we use the labeled portion of [ANTiVax](https://github.com/SakibShahriar95/ANTiVax), randomly split into training and test sets. We also provide an implementation of their BERT-based model for reference.

As the Twitter API rules forbid distributing tweets, the datasets are provided as lists of tweet ID's, which will need to be rehydrated. We recommend using [Hydrator](https://github.com/DocNow/hydrator). Instructions for installation are available in their README file. Run the rehydration tool on dataset-train.txt and dataset-test.txt and input the resulting JSON files into our model.

To Evaluate
-----------
To evaluate the model:

	python eval.py -modelname parameter labels_file training_file

where modelname is rnn or bert, parameter is the amount of L2 regularization or dropout, respectively, labels_file is a file of comma-seperated tweet IDs and labels (1 for misinformation tweets, 0 otherwise), and training_file is a JSON file of tweets obtained by Twitter's API. The ID's of all tweets in training_file must appear in labels_file, but any additional labels will be ignored.

The program will automatically split the data into ten training and test sets (of size 90% and 10% each, respectively) for cross-validation.

Replication
-----------

To replicate the results in the report from our best model using the provided dataset (once the tweets have been downloaded):

	python eval.py -rnn 0.001 dataset/labels.txt datasets/dataset.jsonl
