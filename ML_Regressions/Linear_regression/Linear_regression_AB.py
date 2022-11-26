#dynamic

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
	train_np = readFile(training_data)


# Create input lists----------------------------------------------------------
	l1 = train_np.tolist()





# My code----------------------------------------------------------
	y = [] # array of input y's
	for i in l1:
		y.append(float(i[1]))
	x = [] # array of input x's
	for i in l1:
		x.append(float(i[0]))




# interating K times----------------------------------------------------------
	for k in range(1,11):
		matrix = []
		ys = []
		for d in range(0,7): # Interating d times 
			for datanumber in range(len(l1)):
				if d == 0:
					matrix.append([1])
				else:
					matrix[datanumber].append(math.sin(k*d*float(l1[datanumber][0]))**2) # using formula here if d > 0



			arr1 = np.array(matrix) # converting to np array
			arr1_transpose = arr1.transpose()

			arr2 = np.linalg.inv(np.dot(arr1_transpose,arr1))

			midstep = np.dot(arr2,arr1_transpose) # mid step to find Theta

			tita = np.dot(midstep,np.array(y)) #Theta value for each d and k




# appending ----------------------------------------------------------
			if d == 0:
				ys.append([tita[0]]*len(l1))
			else:
				queue = []
				for j in range(len(l1)):
					summation = 0
					for i in range(1,d+1):
						summation += tita[i]*math.sin(i*k*float(l1[j][0])**2)
					queue.append(tita[0]+summation)


				ys.append(queue)


				

# Plot the graph----------------------------------------------------------
		for d in ys:
			plt.scatter(x,d)

		plt.xlabel("X axis label; K = "+str(k))
		plt.ylabel("Y axis label")
		plt.title("Training Data Size = "+str(len(l1)))
		plt.legend(['d=0','d=1','d=2','d=3','d=4','d=5','d=6'])
		plt.show()




# Main function Call----------------------------------------------------------
if __name__ == "__main__":
    main()