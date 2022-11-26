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
	test_data = '../datasets/Q1_C_test.txt'


# Create input lists----------------------------------------------------------
	train_np = readFile(training_data)
	list1 = train_np.tolist()
	l1 = list1[:20]

	test_np = readFile(test_data)
	l2 = test_np.tolist()




# My code----------------------------------------------------------
	y = []
	for i in l1:
		y.append(float(i[1]))

	x = []
	for i in l1:
		x.append(float(i[0]))




# interating K times----------------------------------------------------------
	for k in range(1,11):
		matrix = []
		ys = [] # when k = 1, ys = [d =1 for all x, d=2 for all x, d =3... ] for
		print("For K = "+str(k)) 
		for d in range(0,7):
			for datanumber in range(len(l1)):
				if d == 0:
					matrix.append([1])
				else:
					matrix[datanumber].append(math.sin(k*d*float(l1[datanumber][0]))**2)


# appending ----------------------------------------------------------
			arr1 = np.array(matrix)
			arr1_transpose = arr1.transpose()

			arr2 = np.linalg.inv(np.dot(arr1_transpose,arr1))

			midstep = np.dot(arr2,arr1_transpose)

			tita = np.dot(midstep,np.array(y))




# Find MSE using its formula when d == 0 and d > 0-----------------------------------
			if d == 0:
				ypredicted = tita[0]
				summation = 0
				for i in l2:
					summation += (ypredicted - float(i[1]))**2
				print('For d = '+str(d)+' MSE = '+str(summation))
				ys.append([tita[0]]*len(l2))
			else:
				mainsummation = 0
				queue = []
				for j in range(len(l2)):
					summation = 0
					for i in range(1,d+1):
						summation += tita[i]*math.sin(i*k*float(l2[j][0])**2)
					ypredicted = tita[0] + summation
					queue.append(ypredicted)
					mainsummation += (ypredicted - float(l2[j][1]))**2
				print('For d = '+str(d)+' MSE = '+str(mainsummation))
		

				ys.append(queue)
		print("------------------")



		
# Find MSE using its formula when d == 0 and d > 0-----------------------------------
		for d in ys:
			plt.scatter(x[:10],d)

		plt.xlabel("X axis label; K = "+str(k))
		plt.ylabel("Y axis label")
		plt.title("Training Data Size = "+str(len(l1)))
		plt.legend(['d=0','d=1','d=2','d=3','d=4','d=5','d=6'])
		plt.show()

	

if __name__ == "__main__":
    main()