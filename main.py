import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Getting data
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())

# Splitting data
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardizing data
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Assessing Feature Importances with Random Forests
feat_labels = df_wine.columns[1:]


# Random Forest Classifier
forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)
forest.fit(X_train, y_train)
print("\n Random forest prediction: ")
y_pred = forest.predict(X_test)
print(y_pred)
print(y_test)
print 'Accuracy score: ', sm.accuracy_score(y_test, y_pred)
print 'Mean squared error: ', sm.mean_squared_error(y_test, y_pred)
print 'MCC: ', sm.matthews_corrcoef(y_test, y_pred)
print("Confusion matrix: ", sm.confusion_matrix(y_test, y_pred))
print("\n")

# K Neighbors Classifier
kn = KNeighborsClassifier(
                            p=2,
                            metric='minkowski')
kn.fit(X_train, y_train)
print("\n KNeighbors prediction: ")
y_pred2 = kn.predict(X_test)
print(y_pred2)
print(y_test)
print 'Accuracy score: ', sm.accuracy_score(y_test, y_pred2)
print 'Mean squared error: ', sm.mean_squared_error(y_test, y_pred2)
print 'MCC: ', sm.matthews_corrcoef(y_test, y_pred2)
print 'Confusion matrix: ', sm.confusion_matrix(y_test, y_pred2)
print("\n")

# Gaussian Classifier
gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
gpc.fit(X_train_std, y_train)
print("\n Gaussian prediction: ")
y_pred3 = gpc.predict(X_test_std)
print(y_pred3)
print(y_test)
print("Accuracy score: ", sm.accuracy_score(y_test, y_pred3))
print("Mean squared error: ", sm.mean_squared_error(y_test, y_pred3))
print("Confusion matrix: ", sm.confusion_matrix(y_test, y_pred3))
print("MCC: ", sm.matthews_corrcoef(y_test, y_pred3))
print("\n")

# Decision Tree Classifier
dtc = DecisionTreeClassifier(max_depth=3)
dtc.fit(X_train_std, y_train)
print("\n Decision Tree prediction: ")
y_pred4 = dtc.predict(X_test_std)
print(y_pred4)
print(y_test)
print("Accuracy score: ", sm.accuracy_score(y_test, y_pred4))
print("Mean squared error: ", sm.mean_squared_error(y_test, y_pred4))
print("Confusion matrix: ", sm.confusion_matrix(y_test, y_pred4))
print("MCC: ", sm.matthews_corrcoef(y_test, y_pred4))
print("\n")

# Neural Network Classifier
mlpc = MLPClassifier(alpha=1e-5, max_iter = 400)
mlpc.fit(X_train_std, y_train)
print("\n Neural Network prediction: ")
y_pred5 = mlpc.predict(X_test_std)
print(y_pred5)
print(y_test)
print("Accuracy score: ", sm.accuracy_score(y_test, y_pred5))
print("Mean squared error: ", sm.mean_squared_error(y_test, y_pred5))
print("Confusion matrix: ", sm.confusion_matrix(y_test, y_pred5))
print("MCC: ", sm.matthews_corrcoef(y_test, y_pred5))
print("\n")

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='crimson',
        align='center')

plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()