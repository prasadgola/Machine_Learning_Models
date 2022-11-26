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
	x = []# X matrix with bias
	for i in l1:
		xarray = [1,float(i[0]), float(i[1]), float(i[2])]
		x.append(xarray)
	y = [0 if i[3] == 'M' else 1 for i in l1]



	def sigmoid(z):
	    return 1 / (1 + np.exp(-z))

	learningrate = 0.01

	m = [0,0,0,0]
	M = np.array(m)
	X = np.array(x)
	Y = np.array(y)
	c = 0




# Find ypredictions----------------------------------------------------------
	yprediction = []

	for i in range(0,20):
		z = np.dot(X, M)

		h = sigmoid(z)
		gradient = np.dot(X.T, (h - Y)) / Y.size
		M = M-(gradient * learningrate)


		ypre = sigmoid(np.dot(X,M))


		TF = ypre > 0.5

		yprediced = [1 if i == True else 0 for i in TF]

		correct = 0

		for u in range(len(y)):
			if y[u] == yprediced[u]:
				correct += 1

		print("Accuracy for int "+str(i+1)+" = "+str((correct/len(y)) *100))




# Plot graph# ----------------------------------------------------------
	givenheight = [i[1] for i in x]
	givenweight = [i[2] for i in x]
	givenage = [i[3] for i in x]
	yprediced = [1 if i == True else 0 for i in TF]




	fig = plt.figure()
	plot = fig.add_subplot(projection = '3d')
	ax = fig.add_subplot(111, projection='3d')



	def get_surface_func(parameter_matrix, x, y):
		bias = parameter_matrix[0]
		h_coeff = parameter_matrix[1]
		w_coeff = parameter_matrix[2]
		a_coeff = parameter_matrix[3]
		z = -1*(bias + h_coeff*x + w_coeff*y)/a_coeff
		return z





	h_vals = np.linspace(1.3, 2.0, 100)
	w_vals = np.linspace(60, 100, 100)
	l, p = np.meshgrid(h_vals, w_vals)
	Z = get_surface_func(M, l, p)#surface plot
	ax.plot_surface(l, p, Z) #surface plot
	ax.scatter(givenheight,givenweight,givenage) 
	plt.show()





if __name__ == "__main__":
    main()