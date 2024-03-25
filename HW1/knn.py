import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


train_data = pd.read_csv("hw1-data/income.train.txt.5k", sep=",", names=["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country", "target"], engine="python")
dev_data = pd.read_csv("hw1-data/income.dev.txt", sep=",", names=["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country", "target"], engine="python")

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Binarization
train_binary_data = encoder.fit_transform(train_data[["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country",]])
dev_binary_data = encoder.transform(dev_data[["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country",]])


num_processor = MinMaxScaler(feature_range=(0, 2))
cat_processor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
preprocessor = ColumnTransformer([("num", num_processor, ["age", "hours"]), ("cat", cat_processor, ["sector", "edu", "marriage", "occupation", "race", "sex", "country"])])

preprocessor.fit(train_data)

train_processed_data = preprocessor.transform(train_data)
dev_processed_data = preprocessor.transform(dev_data)

def myKnnPredict(k, x_train, y_train, x_test, order=2):
    y_pred = []
    for x in x_test:
        distances = np.linalg.norm(x_train - x, ord=order, axis=1)
        k_indices = np.argpartition(distances, k)[:k]
        k_nearest_labels = y_train[k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        y_pred.append(most_common)
    return np.array(y_pred)

X_train = train_processed_data 
y_train = train_data['target'].map({' <=50K': 0, ' >50K': 1})

X_dev = dev_processed_data
y_dev = dev_data['target'].map({' <=50K': 0, ' >50K': 1})

#for 99th
y_pred = myKnnPredict(99, X_train, y_train, X_train)
accuracy = accuracy_score(y_train, y_pred)

y_pred_dev = myKnnPredict(99, X_train, y_train, X_dev)
accuracy_dev = accuracy_score(y_dev, y_pred_dev)

print(f"k = 99 train_err: {(1-accuracy) * 100:.2f}% dev_err: {(1-accuracy_dev) * 100:.2f}%")
