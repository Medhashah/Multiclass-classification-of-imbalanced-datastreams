import random
import datetime
import pickle
import numpy as np
import time
import pandas
import csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from sklearn import model_selection
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier
from collections import defaultdict

from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

dataset = pd.read_csv("./wine-data.csv", sep=";")
dataset.drop('Serial', axis=1, inplace=True)
ln = dataset.__len__()
col = len(dataset.columns)
print(col)

colnames = list(dataset.columns.values)
colnames = colnames[-1:] + colnames[:-1]
print("colnames:", colnames)
className = list(dataset.columns.values)

classN = ';'.join(className)
classNa = classN.split(";")[-1]
print("class name is:", classNa)

group = dataset.groupby(classNa).size()

print(group)
# plt.plot(group)
# plt.ylabel('number of instances')
# plt.xlabel('class description')
# plt.axis('tight')
# plt.show()

classValues = dataset[classNa].unique()

classValues.sort()
print(classValues)

classNumber = len(dataset[classNa].unique())

print("number of classes are: ", classNumber)
# print(classValues[1])
# dataset2 = dataset.set_index(classNa)
# print(dataset.shape)
dataset = dataset.replace('?', '00000')
dataset = dataset.replace('00000', dataset.median())

# print(dataset.shape)
array = dataset.values
X = array[:, 0:col]
Y = array[:, (col - 1)]

validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

colnames = dataset.columns.values
f = open("X_train.csv", "w+")
f2 = open("X_validation.csv", "w+")
colnames = str(colnames)[1:-1].replace(" ", ",").replace("\n", "").replace("'", "\"")
f.write(colnames)
f.write("\n")
f.close()
f2.write(colnames)
f2.write("\n")
f2.close()

# f = open("./X_train.csv", "a")
df = pd.DataFrame(X_train)

df.to_csv("./X_train.csv", sep=",", header=False, index=False, mode='a')
# np.savetxt(f, X_train, delimiter=",", fmt="%d")

# f = open("./X_validation.csv", "a")
df = pd.DataFrame(X_validation)
df.to_csv("./X_validation.csv", sep=",", header=False, index=False, mode='a')
# np.savetxt(f, X_validation, delimiter=",", fmt="%d")
dataset = pd.read_csv("./X_train.csv", sep=",")
print("training dataset is:", dataset.shape)
dataset = pd.read_csv("./X_validation.csv", sep=",")

print("testing dataset is:", dataset.shape)

dataset = pd.read_csv("./X_train.csv", sep=",")
dataset2 = dataset.set_index(classNa)

# making separate files for classes
for c in range(1, classNumber + 1):
    f = open("class data" + str(c) + ".csv", "w+")
    f.close()
    with open('./class data' + str(c) + '.csv', 'a') as f:
        dataset2.loc[float(classValues[(c - 1)]), :].to_csv(f, header=False, sep=",")

classStart = 0
classEnd = 0
colnames = list(dataset.columns.values)
colnames = colnames[-1:] + colnames[:-1]
print("colnames:", colnames)
colnames = str(colnames)[1:-1].replace("\n", "").replace("'", "\"").replace(" ", "")
dataPartClass = colnames

# putting together parts of individual classes
for n in range(1, (classNumber + 1)):
    f = open("data part" + str(n) + ".csv", "w+")
    f.write(str(dataPartClass) + "\n")
    for cn in range(1, classNumber + 1):
        classData = pd.read_csv("./class data" + str(cn) + ".csv", sep=",")
        classLength = len(classData) / classNumber
        classStart = (int(classLength) * (n - 1))
        classEnd = (int(classLength) * n) - 1
        fClass = open("class data" + str(cn) + ".csv", "r")
        lines = fClass.readlines()
        # print("class boundaries", classStart, classEnd)
        # put = str(classStart) + "\n" + str(classEnd) + "\n" + str(cn) + "\n\n"
        # f.write(put)
        # print("lines is: ", lines[int(classLength)])
        for l in range(classStart, classEnd + 1):
            f.write(lines[l])

# print("training")
# print(datetime.datetime.now())
seed = 7
scoring = 'accuracy'
models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
# models.append(('Nu SVM', NuSVC()))
# models.append(('Linear SVM', LinearSVC()))

results = []
names = []
num_trees = 30
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    # mod = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    # results = model_selection.cross_val_score(mod, X, Y, cv=kfold)
    # print(results.mean())
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print(datetime.datetime.now())
# m = 0
# big = results[0].mean()
# for i in range(0, models.__len__()):
#     if results[i].mean() > big:
#         big = results[i].mean()
#         m = i
#     print(results[i].mean(), names[i])
# print(names[m])
# print(models[m][1])

# bestModel = models[m][1]
bestModel = LogisticRegression()
model_number = 1

f = open('./train.csv', 'w+')
f.close()

for model_number in range(1, (classNumber + 1)):
    # print(model_number)
    datasetPart = pd.read_csv("./data part" + str(model_number) + ".csv", sep=",")
    # remove serial

    # dataset.drop('Serial', axis=1, inplace=True)
    print(datasetPart.shape)
    # print(datasetPart.columns.values)
    group = datasetPart.groupby(classNa).size()
    print(group)
    # plt.plot(group)
    # plt.title("data part"+str(model_number))
    # plt.ylabel('number of instances')
    # plt.xlabel('class description')
    # plt.axis('tight')
    # plt.show()

    # labels = classValues
    # sizes = group
    # explode = (0, 0, 0, 0.1, 0, 0, 0)
    # plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=140)
    #
    # plt.axis('equal')
    # plt.show()

    ln = datasetPart.__len__()
    col = len(dataset.columns)
    # print(col, model_number)

    classValues = datasetPart[classNa].unique()
    classValues.sort()
    # print(classValues)

    classNumber = len(datasetPart[classNa].unique())
    print("number of classes are:", classNumber)
    if classNumber == 1:
        break
    array = datasetPart.values

    X = array[:, 0:col]
    Y = array[:, 0]

    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)
    bestModel.fit(X_train, Y_train)

    # f = open("X_train" + str(n) + ".csv", "w+")
    # f.write(X_train)

    # np.savetxt("train.csv", X, delimiter=",")

    predictArray = []
    # Data stream creation and prediction
    predictArrayIndex = 0
    ln = len(X_validation)
    for i in range(0, ln):
        X_stream = X_validation[i:i + 1, 0:col]
        # Y_stream = Y[i:i + 1, (col - 1)]
        # print("X is :", X_stream)
        # print("Y is:", Y_stream)
        delay = random.uniform(0, 2)
        # print("delay is:", delay)
        # time.sleep(delay)
        predictionsStream = bestModel.predict(X_stream)
        predictArray[predictArrayIndex:predictArrayIndex + 1] = predictionsStream
        predictArrayIndex += 1

    filename = 'model_' + str(model_number) + '.sav'
    joblib.dump(bestModel, filename)
    loaded_model = joblib.load(filename)

    # # Dump the trained decision tree classifier with Pickle
    # model_filename = 'model_' + str(model_number) + '.pkl'
    # # Open the file to save as pkl file
    # model_pkl = open(model_filename, 'wb')
    # pickle.dump(bestModel, model_pkl)
    # # Close the pickle instances
    # model_pkl.close()

    # f = open("wine data r" + str((model_number + 1)) + ".csv", "w+")
    # print(colnames)
    # attach = classNa
    # if model_number == 1:
    #     colnames = colnames[1:col]
    # f.write(attach)
    # for j in range(0, len(colnames)):
    #     f.write(";")
    #     f.write(colnames[j])
    #
    # f.write("\n")
    # f.close()

    print("final confusion matrix is: \n", confusion_matrix(Y_validation, predictArray))
    print("final classification report is: \n", classification_report(Y_validation, predictArray))
    print("final accuracy score is: ", accuracy_score(Y_validation, predictArray))
    if accuracy_score(Y_validation, predictArray) == 1:
        print("accuracy achieved.")
        break

    con = classification_report(Y_validation, predictArray).split("\n")
    # print("con is:", con[2])
    # print(len(con))
    dataset2 = datasetPart.set_index(classNa)

    sum = 0
    mini = 1
    maxi = 0
    less = 0
    bigger = 0
    for j in range(2, len(con) - 3):
        con2 = con[j].split(" ")
        print(len(con2))
        if float(con2[21]) < mini:
            mini = float(con2[21])
            less = con2[8]
        if float(con2[len(con2) - 1]) > maxi:
            maxi = float(con2[len(con2) - 1])
            bigger = con2[8]
        sum += float(con2[21])
    print(less, bigger)
    avg = sum / classNumber
    count = 0
    f = open("train" + str(model_number) + ".csv", "w+")
    f.close()

    f = open("sample" + str(model_number) + ".csv", "w+")
    f.close()

    with open('sample' + str(model_number) + '.csv', 'a') as f:
        dataset2.loc[float(less), :].to_csv(f, header=False, sep=",")
        # dataset2.loc[float(bigger), :].to_csv(f, header=False, sep=",")

    # sampleData = pd.read_csv("./train" + str(model_number) + ".csv", sep=",")
    # sampleArray = sampleData.values
    # print(sampleData.shape)
    # X = sampleArray[:, 0:col]
    # Y = sampleArray[:, 0]
    #
    # print('Unsampled dataset shape {}'.format(Counter(Y)))
    # sm = SMOTE(random_state=42, k_neighbors=3)
    # X_res, Y_res = sm.fit_sample(X, Y)
    # print('Resampled dataset shape {}'.format(Counter(Y_res)))
    #
    # np.savetxt("./train" + str(model_number) + ".csv", X_res, delimiter=",")

    for j in range(2, len(con) - 3):
        con2 = con[j].split(" ")
        print(con2[8])
        j += 2
        if (float(con2[21]) < avg) | (float(con2[21]) < 0.8):
            if float(con2[8]) != float(less):
                with open('train' + str(model_number) + '.csv', 'a') as f:
                    dataset2.loc[float(con2[8]), :].to_csv(f, header=False, sep=",")

            print("class tuples added.")
        else:
            print("not required")

    trainData = pd.read_csv("./train" + str(model_number) + ".csv", sep=",")
    trainArray = trainData.values
    print(trainData.shape)
    X = trainArray[:, 0:col]
    Y = trainArray[:, 0]
    print('dataset shape {}'.format(Counter(Y)))
    val = Counter(Y)
    maxiVal = max(val.values())
    print(maxiVal)
    ind = 0
    for classValue in classValues:
        if val[classValue] == maxiVal:
            ind = classValue

    f=open('train' + str(model_number) + '.csv','w+')

    for v in val:
        print(v)
        if v != ind:
            with open('train' + str(model_number) + '.csv', 'a') as f:
                dataset2.loc[v, :].to_csv(f, header=False, sep=",")

    with open('sample' + str(model_number) + '.csv', 'a') as f:
        dataset2.loc[ind, :].to_csv(f, header=False, sep=",")

    sampleData = pd.read_csv("./sample" + str(model_number) + ".csv", sep=",")
    sampleArray = sampleData.values
    print(sampleData.shape)
    X = sampleArray[:, 0:col]
    Y = sampleArray[:, 0]

    print('Unsampled dataset shape {}'.format(Counter(Y)))
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, Y_res = sm.fit_sample(X, Y)
    print('Resampled dataset shape {}'.format(Counter(Y_res)))

    np.savetxt("./sample" + str(model_number) + ".csv", X_res, delimiter=",")

    # sm = SMOTE(random_state=42, k_neighbors=3, ratio='minority')
    # X_res, Y_res = sm.fit_sample(X, Y)
    # print('Resampled dataset shape {}'.format(Counter(Y_res)))
    #
    # np.savetxt("./train" + str(model_number) + ".csv", X_res, delimiter=",")

    sampleData = pd.read_csv("./sample" + str(model_number) + ".csv", sep=",")

    with open('train' + str(model_number) + '.csv', 'a') as f:
        sampleData.to_csv(f, header=False, sep=",", index=False)

    trainData = pd.read_csv("./train" + str(model_number) + ".csv", sep=",")
    # trainData.drop_duplicates(subset=None, inplace=True)

    sampleArray = trainData.values
    print(sampleData.shape)
    X = sampleArray[:, 0:col]
    Y = sampleArray[:, 0]

    print('sampled dataset shape {}'.format(Counter(Y)))

    with open('./train.csv', 'a') as f:
        trainData.to_csv(f, header=False, sep=",", index=False)

    trainData = pd.read_csv("./train.csv", sep=",")
    trainArray = trainData.values
    print(trainData.shape)
    # Xn = trainArray[:, 0:col]
    # Yn = trainArray[:, 0]
    # print('dataset shape {}'.format(Counter(Yn)))
    #
    # sm = SMOTE(random_state=42, k_neighbors=4, ratio='minority')
    # X_res, Y_res = sm.fit_sample(Xn, Yn)
    # print('Resampled dataset shape {}'.format(Counter(Y_res)))
    #
    # np.savetxt("train.csv", X_res, delimiter=",")
    #
    # trainData = pd.read_csv("./train.csv", sep=",")
    # print(trainData.shape)

    with open('./data part' + str((model_number + 1)) + '.csv', 'a') as f:
        trainData.to_csv(f, header=False, sep=",", index=False)

        # datasetPart = pd.read_csv('./data part' + str((model_number + 1)) + '.csv', sep=",")
        # group = datasetPart.groupby(classNa).size()
        # print(group)
        # plt.plot(group)
        # name = 'data part' + str((model_number + 1)) + '.csv'
        # plt.title(name)
        # plt.ylabel('number of instances')
        # plt.xlabel('class description')
        # plt.axis('tight')
        # plt.show()

        # labels = classValues
        # sizes = group
        # explode = (0, 0, 0, 0.1, 0, 0, 0)
        # plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', startangle=140)
        #
        # plt.axis('equal')
        # plt.show()

        # datasetNew = pd.read_csv('./data part' + str((model_number + 1)) + '.csv', sep=",")

        # colna = datasetNew.columns.tolist()
        # print(colna)

        # colna = colna[1:col] + colna[:1]
        # print(colna)
        # # datasetNew = datasetNew[colnames]
        # datasetNew = datasetNew[colna]

        # with open('./wine data r' + str((model_number + 1)) + '.csv', 'w') as f:
        #     datasetNew.to_csv(f, sep=";")
        #
        # with open('./wine data r' + str((model_number + 1)) + '.csv') as fin:
        #     lines = fin.readlines()
        #
        # change = lines[0]
        # print("before change is:", change)
        # lines[0] = lines[0].replace(lines[0], 'Serial' + lines[0])
        # print("change is : ", lines[0])
        # print(lines[1])
        #
        # with open('./wine data r' + str((model_number + 1)) + '.csv', 'w') as fout:
        #     for line in lines:
        #         fout.write(line)

# print("model number is:", model_number)

model = []
model.insert(0, "")
for m in range(1, model_number + 1):
    print(m)
    model.insert(m, joblib.load("model_" + str(m) + ".sav"))

for m in range(1, model_number + 1):
    print(model[m])

# model1 = joblib.load("model_1.sav")
# model2 = joblib.load("model_2.sav")
# model3 = joblib.load("model_3.sav")
# model4 = joblib.load("model_4.sav")
# model5 = joblib.load("model_5.sav")
# model6 = joblib.load("model_6.sav")
# model7 = joblib.load("model_7.sav")

# model_pkl = open('model_1.pkl', 'rb')
# model1 = pickle.load(model_pkl)
# model_pkl = open('model_2.pkl', 'rb')
# model2 = pickle.load(model_pkl)
# model_pkl = open('model_3.pkl', 'rb')
# model3 = pickle.load(model_pkl)
# model_pkl = open('model_4.pkl', 'rb')
# model4 = pickle.load(model_pkl)
# model_pkl = open('model_5.pkl', 'rb')
# model5 = pickle.load(model_pkl)
# model_pkl = open('model_6.pkl', 'rb')
# model6 = pickle.load(model_pkl)
# model_pkl = open('model_7.pkl', 'rb')
# model7 = pickle.load(model_pkl)

# dataset = pd.read_csv("./data part7.csv", sep=",")
# # dataset.drop('Serial', axis=1, inplace=True)
#
# ln = dataset.__len__()
# col = len(dataset.columns)
# print(col)
#
# array = dataset.values
# X = array[:, 0:col]
# Y = array[:, 0]
# model7.fit(X, Y)

dataset = pd.read_csv("./X_validation.csv", sep=",")
# dataset.drop('Serial', axis=1, inplace=True)

ln = dataset.__len__()
col = len(dataset.columns)

colnames = list(dataset.columns.values)
colnames = colnames[-1:] + colnames[:-1]
print("colnames:", colnames)
dataset = dataset[colnames]

print(col)
print(dataset.columns.values)

array = dataset.values

predictArray = []
# predictArray2 = []
# predictArray3 = []
# predictArray4 = []
# predictArray5 = []

for i in range(0, ln):
    X_stream = array[i:i + 1, 0:col]
    Y_stream = array[i:i + 1, 0]
    # print("X is :", X_stream)

    # delay = random.uniform(0, 2)
    #    print("delay is:", delay)
    # time.sleep(delay)
    L = []
    # print("Y is:", Y_stream)
    for j in range(1, model_number + 1):
        predictionsStream = model[j].predict(X_stream)
        L.append(predictionsStream)
        # print("prediction of classifier ", j, ":", predictionsStream)

    # predictionsStream2 = model[2].predict(X_stream)
    # L.append(predictionsStream2)
    # # print("prediction 2:", predictionsStream2)
    # predictionsStream3 = model[3].predict(X_stream)
    # L.append(predictionsStream3)
    # # print("prediction 3:", predictionsStream3)
    # predictionsStream4 = model[4].predict(X_stream)
    # L.append(predictionsStream4)
    # # print("prediction 4:", predictionsStream4)
    # predictionsStream5 = model[5].predict(X_stream)
    # L.append(predictionsStream5)
    # # print("prediction 5:", predictionsStream5)
    # predictionsStream6 = model[6].predict(X_stream)
    # L.append(predictionsStream6)
    # print("prediction 6:", predictionsStream6)
    # predictionsStream7 = model7.predict(X_stream)
    # L.append(predictionsStream7)

    # print(L)
    # print("final classification result:", max(L, key=L.count))

    predictArray[i:i + 1] = max(L, key=L.count)
# predictArray2[i:i + 1] = predictionsStream2
#    predictArray3[i:i + 1] = predictionsStream3
#    predictArray4[i:i + 1] = predictionsStream4
#    predictArray5[i:i + 1] = predictionsStream5

print("final confusion matrix is: \n", confusion_matrix(array[0:ln, 0], predictArray))
print("final classification report is: \n", classification_report(array[0:ln, 0], predictArray))
print("final accuracy score is: ", accuracy_score(array[0:ln, 0], predictArray))
