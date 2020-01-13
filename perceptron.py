import numpy as np
from csv import reader
import matplotlib.pyplot as plt
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


# Make a prediction with weights
def predict(row, weights,threshold):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= threshold else 0.0
 
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch,threshold):
	errorList=[]
	weights = [0  , 0,  0]
	predicted=[]
	for epoch in range(n_epoch):
		l_rate=l_rate-l_rate/n_epoch
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights,threshold)
			if epoch==n_epoch-1:
			   predicted.append(prediction)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        	errorList.append(sum_error)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights,errorList,predicted



# load and prepare data
filename = 'd.txt'
dataset = load_csv(filename)
# convert string class to floats
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)
#normalize dataset
x=[]
y=[]
for row in dataset :
    x.append(row[0])
    y.append(row[1])

for row in dataset:
    row[0]=(row[0]-min(x))/(max(x)-min(x))
    row[1]=(row[1]-min(y))/(max(y)-min(y))
    
# evaluate algorithm
threshold = 2025
l_rate = 0.1
n_epoch = 4000
weights,errorList,predicted = train_weights(dataset, l_rate, n_epoch,threshold)
print(weights)



epoch = np.linspace(1,len(errorList),len(errorList))
plt.subplot(211)
plt.plot(epoch, errorList)
plt.xlabel('Epoch')
plt.ylabel('Sum-of-Squared Error')
plt.title('Perceptron Convergence')

#shape of data
dataset_0_x=[]
dataset_0_y=[]
dataset_1_x=[]
dataset_1_y=[]
for i,row in zip(predicted,dataset):
    if i==1:
        dataset_1_x.append(row[0])
        dataset_1_y.append(row[1])        
    else:
        dataset_0_x.append(row[0])
        dataset_0_y.append(row[1])        
plt.subplot(212)        
plt.scatter(dataset_0_x,dataset_0_y,color='r',label=0)
plt.scatter(dataset_1_x,dataset_1_y,color='b',label=1)      
plt.show()
