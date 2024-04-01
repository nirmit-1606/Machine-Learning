# HW4: Deep Learning for Sentiment Classification

This homework is about using word2vec as features to train (averaged) perceptron (and optionally, 
other ML algorithms such as SVM) for sentiment classification, the same task we did in HW2. But 
instead of sparse features in HW2, we will use dense features from word embeddings here. The point 
of this homework is more about getting familiar with elementary deep learning.

[Complete HW details](https://classes.engr.oregonstate.edu/eecs/fall2023/ai534-400/unit4/hw4/hw4.pdf)

## Word Embeddings

Word embeddings are continuous vector representations of words that capture semantic and syntactic 
relationships between words. They have been shown to significantly improve the performance of many 
natural language processing (NLP) tasks.

## Properties of Word Embeddings

Some interesting properties of word embeddings include:

1. Similar words have similar embeddings: The embeddings of semantically similar words tend to be
close together in the vector space. This can be measured using cosine similarity, which computes the
cosine of the angle between two vectors. For example, the cosine similarity between the embeddings
of the words "cat" and "kitten" would be higher than the cosine similarity between "cat" and "tree".

2. Analogies: Word embeddings can capture analogies, such as "man - woman = king - queen". This can
be illustrated by performing vector arithmetic on the embeddings and finding the closest word in
the vector space. For example:
embedding("man") - embedding("woman") + embedding("queen") ~= embedding("king")

3. Visualizations: Word embeddings can be projected to a lower-dimensional space (e.g., using t-SNE or PCA)
for visualization purposes. This can help in understanding the relationships between words and
identifying clusters of similar words.

## Using Pretrained Word Embeddings in NLP Tasks

Pretrained word embeddings can be used as a starting point for various NLP tasks, such as text classification, 
sentiment analysis, and machine translation. They can be used in the following ways:

1. As input features: The word embeddings can be used as input features for machine learning models, such 
as neural networks or support vector machines. For example, in a text classification task, you could 
average the embeddings of all words in a document to obtain a document-level embedding, which can then 
be used as input to a classifier.

2. As initialization for fine-tuning: In some cases, it might be beneficial to fine-tune the pretrained 
embeddings on a specific task or domain. You can initialize the embedding layer of a neural network 
with the pretrained embeddings and then update the embeddings during training. This can help the model 
to better capture domain-specific knowledge.

3. In combination with other embeddings: Pretrained embeddings can be combined with other types of embeddings, 
such as character-level embeddings or part-of-speech embeddings, to create richer representations for NLP tasks.

By using pretrained embeddings, you can leverage the knowledge captured from large-scale text corpora and 
improve the performance of your NLP models.
