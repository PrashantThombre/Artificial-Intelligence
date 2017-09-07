import csv
import operator
import math
import matplotlib.pyplot as plt
import copy

def getdata():
    #with open("D:\Downloads\KNN Dataset\\banana-10-1tra.dat",'rb') as f:
    f = open('D:\Downloads\KNN Dataset\\banana-10-1tra.dat','r')
    # reader = csv.reader(f)
    count = 0
    train_data = []
    for row in f:
        if (not row.startswith('@')):
            train_data.append([row.rstrip("\n")])
    print (train_data)

    f = open('D:\Downloads\KNN Dataset\\banana-10-1tst.dat')
    # reader = csv.reader(f)
    count = 0
    test_data = []
    for row in f:
        if (not row.startswith('@')):
            test_data.append([row.rstrip("\n")])
    print (test_data)
    return train_data,test_data

def getdata2():
    #with open("D:\Downloads\KNN Dataset\\banana-10-1tra.dat",'rb') as f:
    train_data = []
    test_data = []
    with open("D:\Downloads\KNN_Dataset\\banana-10-1tra.dat", 'rb') as f:
        reader = csv.reader(f)
        train_data_dummy = list(reader)
        train_data_dummy = train_data_dummy[7:]
    for k in train_data_dummy:
        tempList = []
        for i in k:
            j = i.replace(' ', '')
            tempList.append(j)
        train_data.append(tempList)
    #print (train_data)

    with open("D:\Downloads\KNN_Dataset\\banana-10-1tst.dat", 'rb') as f:
        reader = csv.reader(f)
        test_data_dummy = list(reader)
        test_data_dummy = test_data_dummy[7:]
    for k in test_data_dummy:
        tempList = []
        for i in k:
            j = i.replace(' ', '')
            tempList.append(j)
        test_data.append(tempList)
    #print (test_data)
    return train_data,test_data

def euclideanDist(x, xi):
    d = 0.0
    for i in range(len(x) - 1):
        d += pow((float(x[i]) - float(xi[i])),2)  # euclidean distance
    return d


# KNN prediction and model training
def knn_predict(test_data, train_data, k_value):
     for i in test_data:
        eu_Distance = []
        knn = []
        class1 = 0

        class2 = 0
        for j in train_data:
            eu_dist = euclideanDist(i, j)
            eu_Distance.append((j[-1], eu_dist))
        eu_Distance.sort(key=operator.itemgetter(1))
        knn = eu_Distance[:k_value]
        for k in knn:
            if k[0] == '-1.0':
                class1 += 1
            else:
                class2 += 1
        if class1 > class2:
            i.append('-1.0')
        elif class1 < class2:
            i.append('1.0')
        else:
            i.append('NaN')

def accuracy(test_data):
    correct = 0
    for i in test_data:
        if i[-1] == i[-2]:
            correct += 1
    accuracy = float(correct)/len(test_data) *100  #accuracy
    return accuracy


x_list = []
y_list = []
train_data, test_data = getdata2()
for K in xrange(1,52,2):
    print "K = ",K
    train_dataset = copy.deepcopy(train_data)
    test_dataset = copy.deepcopy(test_data)
    knn_predict(test_dataset, train_dataset, K)
    #print test_dataset
    accuracy_value = accuracy(test_dataset)
    print "Accuracy : ",accuracy_value
    x_list.append(K)
    y_list.append(accuracy_value)
print x_list
print y_list
plt.plot(x_list,y_list)
plt.axis([1,K,80,100])
plt.show()
