import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


le = LabelEncoder()
cols = [2,3]
cols_test = [1,2]
# np.set_printoptions(threshold=np.inf)

df = pd.read_csv('./train.csv', delimiter=',', header = None, encoding = 'latin-1', usecols = cols)
df = df.dropna()
data_train = df.to_numpy(dtype='str')
print(data_train)

df = pd.read_csv('./test.csv', delimiter=',', header = None, encoding = 'latin-1', usecols = cols_test)
df = df.dropna()
data_test = df.to_numpy(dtype='str')
print(data_test)



X_train, y_train = data_train[:,0], data_train[:, -1]
X_test, y_test = data_test[:,0], data_test[:, -1]

vectorizer  = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()

classifier = GaussianNB()
classifier.fit(X_train_vectorized, y_train)

y_pred = classifier.predict(X_test_vectorized)
print("ytest",y_test)
print("ypred",y_pred)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display additional evaluation metrics
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Display confusion matrix
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# User input
user_input = input("Enter your text: ")

# Vectorize the user input using the same vectorizer
user_input_vectorized = vectorizer.transform([user_input]).toarray()
print(user_input_vectorized)

# Reshape the input to 2D array
user_input_vectorized = user_input_vectorized.reshape(1, -1)

# Predict the class of the user input
predicted_class = classifier.predict(user_input_vectorized)[0]

# Print the predicted class
print("Predicted Class:", predicted_class)
