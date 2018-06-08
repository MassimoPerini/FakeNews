import sys
sys.path.append("/usr/local/lib/python3.6/site-packages")
import pandas as pd
import numpy as np
import json
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from mlxtend.classifier import StackingClassifier
from sklearn import decomposition
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from RandomForestClassifierWithCoef import RandomForestClassifierWithCoef

SINGLE_FILE_BOOL = False


FILE_TO_ANALYZE = "dataset/final_test_dataset_5.csv"
OUTPUT_FILE_ANALYSIS = "dataset/QDA_prediction_2.json"

SINGLE_FILE = "RESULT.csv"
CSV_TRAIN_FILE = "sentiment_analysis/final_training_dataset_5_threshold_1500.csv"
CSV_TEST_FILE = "sentiment_analysis/final_test_dataset_5.csv"

SEP = "|"
STACKING = False
TARGET_ID_ANALYSIS = True
PCA = False
PCA_DIMENSIONS = 3





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
#8: delta_t
#9: cluster

emotion_array = ["joy", "sadness", "anger", "fear", "disgust"]

COLUMNS_TO_DROP = [0, 1]


if not SINGLE_FILE_BOOL:
    train_data = pd.read_csv(CSV_TRAIN_FILE, header=0, sep=SEP)
    test_data = pd.read_csv(CSV_TEST_FILE, sep=SEP)
    X_train = train_data
    X_test = test_data
    y_train = train_data['fake']
    y_test = test_data['fake']

    X_train = X_train.drop(X_train.columns[COLUMNS_TO_DROP], axis=1)
    X_test = X_test.drop(X_test.columns[COLUMNS_TO_DROP], axis=1)

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    if PCA:
        pca = decomposition.PCA(n_components=PCA_DIMENSIONS)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        print(X_train)
        pca.fit(X_test)
        X_test = pca.transform(X_test)

else:
    data = pd.read_csv(SINGLE_FILE, header=0)
    data = pd.concat([data, pd.get_dummies(data['cluster'], prefix='cluster')], axis=1)
    data.drop(['cluster'], axis=1, inplace=True)
    y = data[["fake"]]
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Dropping the "fear" and the "sentiment" columns, they add noise and no information

#X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.3, random_state=0)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM"#, "Gaussian Process",
         ,"Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

clf1 = KNeighborsClassifier(3)
clf2 = SVC(kernel="linear", C=0.025)
clf3 = SVC(gamma=2, C=1)
clf4 =  GaussianProcessClassifier(1.0 * RBF(1.0)),
clf5 = DecisionTreeClassifier(max_depth=5)
clf6 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#clf7 = MLPClassifier(alpha=1),
clf8 = AdaBoostClassifier(),
clf8 = GaussianNB(),
clf9 = QuadraticDiscriminantAnalysis()

classifiers = [
    #clf1,
    #clf2,
    #clf3,
    #clf5,
    #clf6,
    #clf8,
    clf9]


if STACKING:

    sclf = StackingClassifier(classifiers=[clf4, clf5, clf9],
                              meta_classifier=LogisticRegression())
    scores = model_selection.cross_val_score(sclf, X_train, y_train,
                                             cv=3, scoring='accuracy')
    print(scores)
    sclf.fit(X_train, y_train)
    y_test = np.reshape(y_test.values, (len(y_test), 1))
    y_predict = sclf.predict(X_test)
    print(classification_report(y_test, y_predict))
    matrix = confusion_matrix(y_test, y_predict)
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
        if TARGET_ID_ANALYSIS and index == len(classifiers) - 1:       #QDA

            complete_file = pd.read_csv(CSV_TEST_FILE, sep="|")
            #complete_file.set_index("tweet_id", inplace=True)
            output_file = open(OUTPUT_FILE_ANALYSIS, "w")

            to_dump = []
            correct_predictions = 0
            error_predictions = 0
            for index in range(len(complete_file)):
                target_tweet_id = complete_file["tweet_id"][index]
                target_tweet_index = complete_file.index[complete_file["tweet_id"] == int(target_tweet_id)]
                target_row = np.reshape(X_test[target_tweet_index][0], (1, len(X_test[target_tweet_index][0])))
                prediction = np.asscalar(classifier.predict(target_row))
                #print(X_test[target_tweet_index])
                index_most_sentiment = np.argmax(target_row[0][:len(target_row[0]) - 1])
                emotion_label = emotion_array[index_most_sentiment]
                is_fake = np.asscalar(complete_file.loc[complete_file["tweet_id"] == int(target_tweet_id)]["fake"])
                error = np.abs(int(prediction) - int(is_fake))
                if error == 0:
                    correct_predictions += 1
                else:
                    error_predictions += 1
                print(prediction, is_fake)
                entry = {'tweet_id': str(target_tweet_id), 'predictedId': str(prediction), 'actual_fake': str(is_fake), "emotion": emotion_label}
                to_dump.append(entry)
            json.dump(to_dump, output_file)
            print(error_predictions, correct_predictions)

'''
    train_prediction = []
    test_prediction = []

    for index, classifier in enumerate(classifiers):
        print(names[index])
        classifier.fit(X_train, y_train)
        train_prediction.append(classifier.predict(X_train))
        print('\n\nAccuracy of ', names[index], ' on training set set: {:.2f}'.format(classifier.score(X_train, train_prediction[-1])))

        test_prediction.append(classifier.predict(X_test))


    X_stacking_train = []
    for i in range(len(train_prediction[0])):
        X_prediction_array = []
        for prediction_array in train_prediction:
            X_prediction_array.append(prediction_array[i])
        X_stacking_train.append(X_prediction_array)

    X_stacking_train = np.array([[X_stacking_train[x][y] for y in range(len(X_stacking_train[x]))] for x in range(len(X_stacking_train))])
    print(X_stacking_train)
    print(y_train)

    X_stacking_test = []
    for i in range(len(test_prediction[0])):
        X_prediction_array = []
        for prediction_array in test_prediction:
            X_prediction_array.append(prediction_array[i])
        X_stacking_test.append(X_prediction_array)



    l_reg = MLPClassifier(alpha=1)
    l_reg.fit(X_stacking_train, y_train)
    y_predict = l_reg.predict(X_stacking_test)
    print(classification_report(y_test, y_predict))
    print('\n\nAccuracy of Logistic Regression on test set: {:.2f}'.format(l_reg.score(X_stacking_test, y_test)))
    matrix = confusion_matrix(y_test, y_predict)
    print(matrix)
    y_pred = []
    for index in range(len(y_test)):
        #print(index, len(y_test))
        x_sample = X_test[index].reshape((1, X_test.shape[1]))
        counter = 0
        #print("VALUES")
        for classifier in classifiers:
            value = classifier.predict(x_sample)[0]
            #print(value)
            if value:
                counter += 1
        predicted_value = round(counter/len(classifiers))
        #print(predicted_value, y_test.values[index])
        if predicted_value == 1:
            y_pred.append(True)
        else:
            y_pred.append(False)

    print(classification_report(y_test, y_pred))
    print("\n\nCONFUSION MATRIX:\n")
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
'''