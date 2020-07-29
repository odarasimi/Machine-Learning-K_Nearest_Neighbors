import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
import matplotlib.pyplot as plt
from matplotlib import style
from math import sqrt
import warnings

'''
df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace=True)
df.drop(["id"],1,inplace=True)

x = np.array(df.drop(["class"],1))
y = np.array(df["class"])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
accuracy = (clf.score(x_test,y_test))

def predict(data):
	if type(data[0]) is list:
		data = data.reshape(len(test), -1)
	else:
		data = data.reshape(1, -1)
	outcome = clf.predict(data)
	if outcome == 4:
		print ("likely malignant")
	else:
		print ("likely benign")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++		

#without a classifier
style.use("fivethirtyeight")
dataset = {"k": ([1,2],[2,3],[3,1]), "r":([6,5],[7,7],[8,6])}
predict = [5,7]

#[plt.scatter(a[0],a[1]) for i in data for a in data[i]]
#plt.show()

def k_nearest_neighbours(data, predict, k=3):
	if len(data) >= k:
		warnings.warn("eww")
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append((euclidean_distance, group))
	
	votes = sorted(distances)[:k][0][1]
	return (votes)
print(k_nearest_neighbours(dataset, predict))

[plt.scatter(a[0],a[1]) for i in dataset for a in dataset[i]]
plt.scatter(predict[0],predict[1])
plt.show() 
'''

def k_nearest_neighbours(data, predict, k=3):
	if len(data) >= k:
		warnings.warn("eww")
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append((euclidean_distance, group))
	
	votes = sorted(distances)[:k][0][1]
	return (votes)

df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace=True)
df.drop(["id"], 1, inplace=True)
full_data = df.astype(float).values.tolist()
#full_data_shuff = random.shuffle(full_data)
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
	train_set[i[-1]].append(i[-1])

for i in test_data:
	test_set[i[-1]].append(i[-1])

correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		return1 = k_nearest_neighbours(train_set, data, k = 5)
		if return1 == group:
			correct += 1
		total += 1	

print ("Accuracy is {} %".format(float(correct)/float(total)*100.0))





