# HW2: Perceptron for Sentiment Classification

This homework is about sentiment classification, using (averaged) perceptron (and other ML algorithms such as SVM). 
Prof. Liang Huang provided a very simple naive perceptron baseline. The point of this homework is more about learning 
from his code and extending it, rather than implementing by oneself from scratch.

[Complete HW details](https://classes.engr.oregonstate.edu/eecs/fall2023/ai534-400/unit2/hw2/hw2.pdf)

## Classification

In classification, you train a machine learning model to classify an input object (could be an image, 
a sentence, an email, or a person described by a group of features such as age and occupation) into 
two or more classes. The very basic (and most widely used) setting is "binary classification" where 
the output label is binary, e.g., whether an image contains a human, whether an email is spam, whether 
a review is positive or negative, or whether a person (based on his/her features) is likely to earn 
more than 50K annually. On the other hand, "multiclass classification" classifies an input object into
one of the many classes; for example, handwritten digit classification (ten classes, 0-9) is widely used 
by USPS, and intent classification (need help, complaint, general inqury, etc.) is widely used in dialog 
systemes deployed in customer service.

## Perceptron Algorithm

The training procedure of the perceptron algorithm is extremely simple. It could be summarized by the following steps:

- initialize weight vector w
- cycle through the training data (multiple iterations)
  - update w if there is a mistake on example (x,y)
- until all examples are classified correctly
