# HW3: Kaggle Competition for Housing Price Prediction

In this HW, I competed in an active Kaggle competition, House Prices - Advanced Regression Techniques.

[Complete HW details](https://classes.engr.oregonstate.edu/eecs/fall2023/ai534-400/unit3/hw3/hw3.pdf)

## Regression

Regression is a type of supervised learning in machine learning that aims to predict continuous numerical values 
based on input data. It involves finding the relationship between input features and output value, which is 
typically represented as a straight line or curve that best fits the data. Regression models are commonly 
used in a variety of fields, such as finance, economics, and engineering, to make predictions on future trends 
or events. We can apply regression analysis to predict stock prices, company revenue and wind speed, as long as 
there are suitable features correlated to the targeted variable. The goal of regression is to minimize the 
difference between the predicted values and actual values, which is typically measured using a loss function 
such as mean squared error.

## Overfitting and Underfitting

In machine learning, the goal is to build models that can generalize well to new data. However, sometimes models 
can either be too simple or too complex, leading to underfitting or overfitting, respectively. Underfitting occurs 
when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on the 
training data as well as on new data. On the other hand, overfitting occurs when a model is too complex and fits 
the training data too closely, resulting in excellent performance on the training data but poor performance on 
new data. Overfitting is a common problem in machine learning, especially when working with high-dimensional data 
or limited amounts of training data.

## Regularization

Regularization is a technique that can be used to prevent overfitting in linear regression models and improve the 
generalization performance. Regularization works by adding a penalty term to the cost function that the linear 
regression model is trying to minimize. The penalty term is based on the magnitude of the weight (or coefficients) 
in the model. By adding this penalty term, the model is encouraged to use smaller coefficients, which results in 
a simpler model that is less likely to overfit the data.
