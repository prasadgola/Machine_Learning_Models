def main():



# Import files----------------------------------------------------------
	import numpy as np
	import math
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

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
	training_data = '../datasets/Q3_data.txt'

# Create input lists----------------------------------------------------------
	train_np = readFile(training_data)
	l1 = train_np.tolist()



# My code----------------------------------------------------------
	x = []
	for i in l1:
		xarray = [1,float(i[0]), float(i[1])]
		x.append(xarray)

	y = [0 if i[3] == 'M' else 1 for i in l1]



	def sigmoid(z):
	    return 1 / (1 + np.exp(-z))
	  
	learningrate = 0.01

	m = [0,0,0]
	M = np.array(m)
	c = 0




# Find ypredictions----------------------------------------------------------
	yprediction = []

	correct = 0

	for g in range(len(x)):
		for v in range(len(x)):
			if g == v:
				continue
			X = np.array(x[:v]+x[v+1:])
			Y = np.array(y[:v]+y[v+1:])
			for i in range(0,20):
				z = np.dot(X, M)

				h = sigmoid(z)
				gradient = np.dot(X.T, (h - Y)) / Y.size
				M = M-(gradient * learningrate)


			ypre = sigmoid(np.dot(X,M))


			TF = ypre > 0.5

			yprediced = [1 if i == True else 0 for i in TF]




# Calculate correct predections---------------------------------------------------------
			for u in range(len(y)-1):
				if y[u] == yprediced[u]:
					correct += 1


# Print out ----------------------------------------------------------
	print("")

	print("Height, Weight Only")

	print("For Alpha = "+str(0.01)+(",intertions = 20"))

	print("Accuracy for int "+str(i+1)+" = "+str((correct/(len(y)**3)) *100)+"%")

	print("")


if __name__ == "__main__":
    main()
    