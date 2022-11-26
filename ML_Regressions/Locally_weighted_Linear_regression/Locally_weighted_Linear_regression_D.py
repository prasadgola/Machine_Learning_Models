def main():



# Import files----------------------------------------------------------
	import numpy as np
	import math
	import matplotlib.pyplot as plt

	def clean_data(line):
	    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

	def fetch_data(filename):
	    with open(filename, 'r') as f:
	        input_data = f.readlines()
	        clean_input = list(map(clean_data, input_data))
	        f.close()
	    return clean_input

	def readFile(dataset_path):
	    input_data = fetch_data(dataset_path)
	    input_np = np.array(input_data)
	    return input_np


# Input files----------------------------------------------------------
	training_data = '../datasets/Q1_B_train.txt'



# Create input lists----------------------------------------------------------
	train_np = readFile(training_data)
	l1 = train_np.tolist()[:20]




# My code----------------------------------------------------------
	y = []
	for i in l1:
		y.append(float(i[1]))
	x = []
	for i in l1:
		x.append(float(i[0]))
	X = np.array(x)



# calculating weight using formula given in pdf----------------------------------------------------------
	w = np.array([np.exp(- (X - x[i])**2/(2*0.204)) for i in range(len(x))])


# Finding Theta values using and Yprediction ----------------------------------------------------------
	ypredictedset = np.zeros(len(l1)) # create zeros for the length

	for i in range(len(l1)):
	    weights = w[:, i]
	    b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
	    A = np.array([[np.sum(weights), np.sum(weights * x)],[np.sum(weights * x), np.sum(weights * x * x)]])
	    theta = np.linalg.solve(A, b)
	    ypredictedset[i] = theta[0] + theta[1] * x[i]



# Formula for MSE----------------------------------------------------------
	summation = 0

	for yp, oy in zip(ypredictedset, y):
		summation += (yp - oy)**2


	print("Data size = 20, MSE = "+ str(summation/len(y)))




# Plot graph----------------------------------------------------------
	plt.scatter(x,ypredictedset)
	plt.show()



if __name__ == "__main__":
    main()
    