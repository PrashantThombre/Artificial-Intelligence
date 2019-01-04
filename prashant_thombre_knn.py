#######################################################################################################################
#   This implementation is to understand how KNN works in practice for simple binary classification.
#   The datasets are from KEEL.
#
#   
#   Author: Prashant Thombre
#   Usage: python prashant_thombre_knn.py <value_of_k> <folder_name>
#           value_of_k: odd integers 1,3,5... etc. [OR] "default" to iterate over K = 1 to 10 and then plot graph in a PDF
#           folder_name: Name of the folder containing the files
#
#   Note: Tested the execution of this program on "banana-10-fold" and "led7digit-10-fold"
#   Sample Run Command: python prashant_thombre_knn.py default banana-10-fold
#   Execution Time for default is more
########################################################################################################################

import csv
import operator
import math
import matplotlib.pyplot as plt
import copy
import sys
import os
from matplotlib.backends.backend_pdf import PdfPages

class Agent:
    def __init__(self,k):
        self.k_value = k

#Sensor part receives test and train data from the environment
    def sensor(self,test_dataset, train_dataset,k_value):
        self.knn_predict(test_dataset, train_dataset,k_value)

#Function to calculate the distance between two data points
    def euclideanDist(self,x, xi):
        d = 0.0
        for i in range(len(x) - 1):
            d += pow((float(x[i]) - float(xi[i])),2)  # euclidean distance calculation without the square root value
        return d

    #The function component to run actual KNN logic
    def knn_predict(self,test_data, train_data, k_value):
        # For each row in test data compute the distance from all the rows in train data
        for i in test_data:
            eu_Distance = []
            knn = []

            for j in train_data:
                eu_dist = self.euclideanDist(i, j)
                eu_Distance.append([j[-1], eu_dist])        #Append the class and distance in a list of lists
            eu_Distance.sort(key=operator.itemgetter(1))    #Sort this list of lists based on the distance
            knn = eu_Distance[:k_value]                     #select top K elements from the list
            i.append(self.action(knn))                      # Pass the filtered list to the actuator component

#Actuator component to classify each percept based on the K-nearest neighbors
#This function returns the most likely label back to the environment
    def action(self,knn):
        temp1 = [x[0] for x in knn]
        temp2 = [[x, temp1.count(x)] for x in set(temp1)]
        knnDict = dict(temp2)
        return max(knnDict.iteritems(), key=operator.itemgetter(1))[0]


#Environment Class Begins Here
class Environment:

#Initializing
    def __init__(self):
        global agent
        agent = Agent(3)

    def run(self):
        x_list = []
        y_list = []
        K = []
        folder_name = sys.argv[2]
        if (sys.argv[1] != "default"):
            K.append(int(sys.argv[1]))
        else:
            K = xrange(1, 12, 2)

        average_accuracy_list = []
        for k_val in K:
            print "\nK = ", k_val
            accuracy_list = []
            for itr in xrange(1, 11):
                train_data, test_data = self.getdata2(folder_name, itr)
                train_dataset = copy.deepcopy(train_data)
                test_dataset = copy.deepcopy(test_data)
                agent.sensor(test_dataset, train_dataset,k_val)
                # test_dataset list has the most likely label returned back from the agent
                # So we can use that list to calculate the accuracy value
                accuracy_value = self.accuracy(test_dataset)
                accuracy_list.append(accuracy_value)
                print "Accuracy : ", accuracy_value
            average_accuracy_list.append(sum(accuracy_list) / len(accuracy_list))
            print "Average Accuracy over 10 files : ",average_accuracy_list[-1]

        if(sys.argv[1] == "default"):
            print "\nList of average accuracies for varying K values"
            print average_accuracy_list
            plt.plot(K,average_accuracy_list)
            plt.axis([1,K[-1],0,100])
            #plt.show()
            #Code to save the plot in a PDF file
            pp = PdfPages('prashant_thombre_plot.pdf')
            plt.savefig(pp, format='pdf')
            pp.close()

#Function to receive the input files
    def getdata2(self,folder_name,itr):
        print (folder_name+" "+str(itr))
        train_data = []
        test_data = []
        count = 0
        #Extract the name for files in the dataset. Line 1 is required if the path is absolute
        name = folder_name.split(os.sep)
        name = name[-1].split("-")
    #Compute the number of rows in the file to skip: The lines starting with @ sign
        with open(folder_name+os.sep+name[0]+"-10-"+str(itr)+"tra.dat", 'rb') as f:
            for line in f.readlines():
                if(line.startswith("@")):
                    count = count+1
                else:
                    break
    #Read the training file and store it in a list
        with open(folder_name+os.sep+name[0]+"-10-"+str(itr)+"tra.dat", 'rb') as f:
            read_file = csv.reader(f)
            train_data_dummy = list(read_file)
            train_data_dummy = train_data_dummy[count:]
        for k in train_data_dummy:
            #Removing the white spaces in each element to avoid errors in comparision
            tempList = []
            for i in k:
                j = i.replace(' ', '')
                tempList.append(j)
            train_data.append(tempList)
        #print (train_data)

    #Read the corresponding testing file and store it in a separate list
        with open(folder_name+os.sep+name[0]+"-10-"+str(itr)+"tst.dat", 'rb') as f:
            reader = csv.reader(f)
            test_data_dummy = list(reader)
            test_data_dummy = test_data_dummy[count:]
        for k in test_data_dummy:
            # Removing the white spaces in each element to avoid errors in comparision
            tempList = []
            for i in k:
                j = i.replace(' ', '')
                tempList.append(j)
            test_data.append(tempList)
        #print (test_data)
        return train_data,test_data

    def accuracy(self,test_data):
        correct = 0
    #Accuracy is calculated based on the returned labels from the actuator. Each row in test_data is appended with the predicted class label
    #And each row has the actual
        for i in test_data:
            if i[-1] == i[-2]:
                correct += 1
        accuracy = float(correct)/len(test_data) *100  #accuracy
        return accuracy

#Entry Point of the program->
if __name__ == "__main__":
    if(len(sys.argv)!=3):
        print "Usage: python prashant_thombre_knn.py <value_of_k> <folder_name>"
        print "\n\tvalue_of_k:\n\tPlease enter an odd integer e.g. 1,3,5.. etc. \n\t[OR]"
        print "\tEnter \"default\" to vary K over odd values between 1 to 10"
        print "\n\tfolder_name: Name of the folder containing the DAT files.\n\tPlease make sure that the folder name does not have any trailing slashes(/ or \) for correct execution."
        print "\nSample Run Command: python prashant_thombre_knn.py 5 banana-10-fold"
    else:
        if(sys.argv[1] != "default"):
            try:
                int(sys.argv[1])
            except:
                print "Please enter an integer value for K"
                print "Terminating the program."
                exit(1)

        obj1 = Environment()
        obj1.run()
