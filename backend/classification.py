import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#Columns: tweet_id, fake, joy, sadness, anger, fear, disgust, sentiment

data = pd.read_csv("./tweets_dataset.csv", header=0, sep=", ")
print(data.columns)

#Dropping the "fear" and the "sentiment" columns, they add noise and no information
X = data.drop(data.columns[[5, 7]], axis=1)
y = data['fake']

print("VALUE COUNTS:")
print(data['fake'].value_counts())


#print(data.groupby('fake').mean())     #to check the mean for every column

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]



for index, classifier in enumerate(classifiers):

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print('\n\nAccuracy of ', names[index], ' on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))

    print("\n\nCONFUSION MATRIX:\n")
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)


