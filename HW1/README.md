# Nearest Neighbor Classifier

The nearest neighbor classifier is a simple machine learning algorithm that is often used for classification tasks. 
It works by finding the "nearest" training example (i.e., the one with the smallest distance) to a new, unlabeled 
example in the feature space, and assigning the label of that nearest example to the new example.  
  
In other words, the nearest neighbor classifier makes predictions based on the similarity of a new data point to the 
data points in the training set. The algorithm is called "lazy" because it does not actually learn a model from the 
training data; instead, it simply memorizes the training examples and uses them to make predictions at test time.  
  
<img style="display: block; margin-left: auto; margin-right: auto;" title="" src="https://upload.wikimedia.org/wikipedia/commons/e/e7/KnnClassification.svg" alt="k-NN">

## HW1: k-NN for Income Prediction

HW1 is about data-preprocessing and k-nearest neighbor classification. You will learn to use many tools, including numpy, 
scikit-learn (sklearn), and basic Unix/Linux/MacOS terminal commands. You will also learn to use sklearn for binarization 
and kNN, and implement an efficient kNN classifier in numpy.

[Complete HW details](https://classes.engr.oregonstate.edu/eecs/fall2023/ai534-400/unit1/hw1/hw1_v2.pdf)
