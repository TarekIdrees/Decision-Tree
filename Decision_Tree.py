# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

#reading Data
path = "house-votes-84.data"
data=pd.read_csv(path,names=['Class Name','handicapped-infants','water-project-cost-sharing'
                             ,'adoption-of-the-budget-resolution','physician-fee-freeze',
                             'el-salvador-aid','religious-groups-in-schools','anti-satellite-test-ban',
                             'aid-to-nicaraguan-contras','mx-missile','immigration','synfuels-corporation-cutback'
                             ,'education-spending','superfund-right-to-sue','crime','duty-free-exports','export-administration-act-south-africa'],
                             na_values=["?"])
colNames = list(data.columns)
#replace missing vlue with most frequent value 
data.replace('?', np.NaN)
for i in colNames:
    data[i] = data[i].fillna(data[i].mode()[0])


#Encoding Categorical data to numeric value
data2 = data
for i in colNames:
    #creating label encoder
    le = preprocessing.LabelEncoder()
    #convert string into numbers
    data2[i] = le.fit_transform(data2[i])

#fetarues
X =data2.iloc[ : ,1:]
#class
y =data2.iloc[ : ,0:1]

testSize = [0.6,0.5,0.4,0.3,0.2]
maxAccuracy = 0
accuracyList = []
bestCLF = None
iterationNumber = 0
testSizeIndex = 0
treeNodesCount=[]
for ieration in range(1,4):
    List = []
    nodes=[]
    for i in range(5):
        #split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize[i])
        #create calssifier
        clf = tree.DecisionTreeClassifier()
        #fit the data
        clf = clf.fit(X_train,y_train)
        #predict test data
        y_predict = clf.predict(X_test)
        List.append(metrics.accuracy_score(y_test, y_predict))
        nodes.append(clf.tree_.node_count)
        if List[i] > maxAccuracy:
            maxAccuracy = List[i]
            bestCLF = clf
            iterationNumber = ieration
            testSizeIndex = testSize[i]
        print("Accuracy with : ", testSize[i]," and  run time number : ",ieration," = ",List[i])
    print("Max Accuracy of ",ieration," st Time = ",max(List))
    print("Min Accuracy of ",ieration," st Time = ",min(List))
    print("Mean Accuracy of ",ieration,"'st Time = ",sum(List)/len(List))
    print("Max Size of ",ieration,"'st Time = ",max(nodes))
    print("Min Size of ",ieration,"'st Time = ",min(nodes))
    print("Mean Size of ",ieration,"'st Time = ",int(sum(nodes)/len(nodes)))
    accuracyList.append(List)
    treeNodesCount.append(nodes)
    print("---------------------------------------------------------")
print("The max accuracy of iteration:",iterationNumber," , testSize: ",testSizeIndex," = ",maxAccuracy)

#plot Nodes number against Accuracy of final tree iteration
SizesAndAccurcy = []
z = 0
for i in testSize:
    List = []
    List.append(str(i))
    for j in range(3):
        List.append(accuracyList[j][z])
    z += 1
    SizesAndAccurcy.append(List)
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(treeNodesCount[iterationNumber-1],accuracyList[iterationNumber-1])
ax.set_xlabel('Nodes Number')
ax.set_ylabel('Accuracy')
ax.set_title('Nodes Number vs Accuracy')
plt.show()

#plot Accuracy against test size of each iteration
df = pd.DataFrame(SizesAndAccurcy, columns=['Test Size', 'Itreation 1', 'Itreation 2', 'Itreation 3'])
df.plot(x="Test Size", y=['Itreation 1', 'Itreation 2', 'Itreation 3'],
        kind="line", figsize=(10, 10))
plt.show()

#plot the decission tree of best calssifier
import graphviz
dot_data = export_graphviz(bestCLF, out_file=None,
                filled=True, rounded=True,
                special_characters=True, feature_names = colNames[1:],class_names=['democrat','republican'])
graph = graphviz.Source(dot_data)
graph.render("DessionTree.png")