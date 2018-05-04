import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


CSV_TRAIN_FILE = "trainin_final_event.csv"
CSV_TEST_FILE = "testing_final_event.csv"
SEP = ","
ENSEMBLES = False


def bootstrap(X_train, y_train, train_array, test_array):
    x_range = np.arange(X_train.shape[0])
    new_indexes = np.random.choice(x_range, size=X_train.shape[0], replace=False)
    bootstrapped_train = X_train[[new_indexes]]
    bootstrapped_test = y_train[[new_indexes]]
    train_array.append(bootstrapped_train)
    test_array.append(bootstrapped_test)


#Columns
#0: tweet_id    must be dropped
#1: fake        must be dropped
#2:  joy
#3: sadness
#4: anger
#5: fear
#6: disgust
#7: sentiment
COLUMNS_TO_DROP = [0, 1, 6, 7]



train_data = pd.read_csv(CSV_TRAIN_FILE, header=0, sep=SEP)
test_data = pd.read_csv(CSV_TEST_FILE, sep=",")
X_train = train_data
X_test = test_data

print("TRAINING DATA MEAN")
print(train_data.groupby('fake').mean())
print("\nTESTING DATA MEAN")
print(test_data.groupby('fake').mean())

#Dropping the "fear" and the "sentiment" columns, they add noise and no information
X_train = train_data.drop(train_data.columns[COLUMNS_TO_DROP], axis=1)
X_test = test_data.drop(test_data.columns[COLUMNS_TO_DROP], axis=1)

y_train = train_data['fake']
y_test = test_data['fake']

print("TRAIN VALUE COUNTS:")
print(train_data['fake'].value_counts())

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
#X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.3, random_state=0)

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

if ENSEMBLES:
    bootstrapped_training_dataset = []
    bootstrapped_testing_dataset = []
    for classifier in range(len(classifiers)):
        bootstrap(X_train, y_train.values, bootstrapped_training_dataset, bootstrapped_testing_dataset)

    for index, classifier in enumerate(classifiers):
        classifier.fit(bootstrapped_training_dataset[index], bootstrapped_testing_dataset[index])

    y_pred = []
    for index in range(len(y_test)):
        #print(index, len(y_test))
        x_sample = X_test[index].reshape((1, X_test.shape[1]))
        counter = 0
        print("VALUES")
        for classifier in classifiers:
            value = classifier.predict(x_sample)[0]
            print(value)
            if value:
                counter += 1
        predicted_value = round(counter/len(classifiers))
        print(predicted_value, y_test.values[index])
        if predicted_value == 1:
            y_pred.append(True)
        else:
            y_pred.append(False)

    print(classification_report(y_test, y_pred))
    print("\n\nCONFUSION MATRIX:\n")
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

else:
    for index, classifier in enumerate(classifiers):

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        print('\n\nAccuracy of ', names[index], ' on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
        print(classification_report(y_test, y_pred))

        print("\n\nCONFUSION MATRIX:\n")
        matrix = confusion_matrix(y_test, y_pred)
        print(matrix)

