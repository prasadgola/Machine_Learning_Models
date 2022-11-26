def main():
    import numpy as np
    import math



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



    training_data = './datasets/Q1_train.txt'
    train_np = readFile(training_data)
    root = train_np.tolist()


    test_data = './datasets/Q1_test.txt'
    test_np = readFile(test_data)
    test = test_np.tolist()

    featurevalues = ['height', 'weight', 'age']





# Entropy -------------------------------------------------------------------------------------------------------------------

    def entropy(array):
        m,w = 0,0
        for i in array:
            if i[3] == 'M':
                m += 1
            else:
                w += 1

        if m == 0:
            return 0
        elif w == 0:
            return 0
        else:
            return ( - (( m/len(array) ) * math.log2(m/len(array)) + ( w/len(array) ) * math.log2(w/len(array))) )



#Funtion define
    def createnode(listc,thresholdval, feature, value):
        return [listc,thresholdval, feature, value]




#create a tree ------------------------------------------------------------------------------------------------------
    class TreeNode:
        def __init__(self, val: int, left=None, right=None) -> None:
            self.val = val
            self.left = left
            self.right = right

        def __repr__(self) -> str:
            return f"val: {self.val}, left: {self.left}, right: {self.right}"

        def __str__(self) -> str:
            return str(self.val)

    def to_binary_tree(items):
        n = len(items)
        if n == 0:
            return None

        def inner(index: int = 0) -> TreeNode:
            """Closure function using recursion bo build tree"""
            if n <= index or items[index] is None:
                return None

            node = TreeNode(items[index])
            node.left = inner(2 * index + 1)
            node.right = inner(2 * index + 2)
            return node

        return inner()






# BFS -----------------------------------------------------------------------------------------------------------------------------
    level = [root]
    depth = 0
    nodesinlists = []

    while level:
        queue = []
        depth += 1

        # base condition ------------------------------------------------------------------------------------------------
        if depth > 5:
            break





        count = 0
        #level interation -----------------------------------------------------------------------------------------------
        for l1 in level:
            count += 1

            if l1 == None:
                queue.append(None)
                queue.append(None)
                nodesinlists.append(createnode(None,None,None,None))
                continue
            if entropy(l1) == 0.0:
                nodesinlists.append(createnode(None,None,None, l1[0][3]))
                queue.append(None)
                queue.append(None)
                continue







            ig = []
            threshold = []

        #Calculating information gain and threshold for all features for current node

            # IG/Threshold based on Height --------------------------------------------------------------------------------
            l1.sort(key=lambda x:x[0])
            for i in range(len(l1)-1):
                left = l1[:i+1]
                right = l1[i+1:]
                left_entropy = entropy(left)
                right_entropy = entropy(right)
                avg_weighted_entropy = ( (len(left)/len(l1)) * left_entropy) + ( (len(right)/len(l1)) * right_entropy )

                threshold.append( ( float(left[len(left)-1][0]) + float(right[0][0]) ) / 2 )
                ig.append(entropy(l1) - avg_weighted_entropy)







            # sorting according to weight -------------------------------------------------------------------------------
            l1.sort(key=lambda x:x[1])
            for i in range(len(l1)-1):
                left = l1[:i+1]
                right = l1[i+1:]
                left_entropy = entropy(left)
                right_entropy = entropy(right)

                avg_weighted_entropy = ( (len(left)/len(l1)) *left_entropy) + ( (len(right)/len(l1)) *right_entropy)
                ig.append(entropy(l1)-avg_weighted_entropy)
                threshold.append( ( float(left[len(left)-1][1]) + float(right[0][1]) ) / 2 )

            # sorting according to age ---------------------------------------------------------------------------------
            l1.sort(key=lambda x:x[2])
            for i in range(len(l1)-1):
                left = l1[:i+1]
                right = l1[i+1:]
                left_entropy = entropy(left)
                right_entropy = entropy(right)

                avg_weighted_entropy = ( (len(left)/len(l1)) *left_entropy) + ( (len(right)/len(l1)) *right_entropy)
                ig.append(entropy(l1)-avg_weighted_entropy)
                threshold.append( ( float(left[len(left)-1][2]) + float(right[0][2]) ) / 2 )






        # Finding index for max ig  ----------------------------------------------------------------------------------------------
            index = ig.index(max(ig))


        # Getting Threshold, feature value using index of max IG -----------------------------------------------------------------------
            if index in range(len(threshold)//3):
                feature = 0
                thresholdvalue = threshold[index]
            elif index in range(len(threshold)//3, (len(threshold)//3)*2):
                feature = 1
                thresholdvalue = threshold[index]
                index =  index - len(threshold)//3
            else:
                feature = 2
                thresholdvalue = threshold[index]
                index = index - (len(threshold)//3)*2

            nodesinlists.append(createnode(l1,thresholdvalue,feature,None))

        #spliting--------------------------------------------------------------------

            l1.sort(key=lambda x:x[feature])
            gender = l1[index][3]
            leftchild = []
            rightchild = []

            for i in l1:
                if float(i[feature]) <= thresholdvalue:
                    leftchild.append(i)
                else:
                    rightchild.append(i)

            queue.append(rightchild)
            queue.append(leftchild)





        rootnode = to_binary_tree(nodesinlists)


        head = rootnode

        #travering all test data in two conditions voting and no voting

        test_op = []

        for i in test:

            temp_depth = 1

            while True:

                # if you get m or w, just append
                if head.val[3] != None:
                    # print('1', head.val[3])
                    test_op.append(head.val[3])
                    break

                if temp_depth == depth:
                    # print("2", 'vote')
                    # vote here and store in test_op
                    vcountm = 0
                    vcountw = 0
                    for i in head.val[0]:
                        if i[3] == "M":
                            vcountm += 1
                        else:
                            vcountw += 1

                    if vcountw > vcountm:
                        test_op.append('W')
                    else:
                        test_op.append('M')


                    break

                if float(i[int(head.val[2])]) < head.val[1]:
                    head = head.left
                else:
                    head = head.right

                temp_depth += 1

            head = rootnode

            



        # test_op.append(head.val[2])

        head = rootnode
        testcorrect = 0
        for i in range(len(test_op)):
            if test_op[i] == test[i][3]:
                testcorrect += 1


    #travering all test data in two conditions voting and no voting


        head = rootnode


        train_op = []

        traindata = train_np.tolist()

        for i in traindata:

            temp_depth = 1

            while True:

                # if you get m or w, just append
                if head.val[3] != None:
                    # print('1', head.val[3])
                    train_op.append(head.val[3])
                    break

                if temp_depth == depth:
                    # print("2", 'vote')
                    # vote here and store in test_op
                    vcountm = 0
                    vcountw = 0
                    for i in head.val[0]:
                        if i[3] == "M":
                            vcountm += 1
                        else:
                            vcountw += 1

                    if vcountw > vcountm:
                        train_op.append('W')
                    else:
                        train_op.append('M')


                    break

                if float(i[int(head.val[2])]) < head.val[1]:
                    head = head.left
                else:
                    head = head.right

                temp_depth += 1

            head = rootnode

            




        head = rootnode
        traincorrect = 0
        for i in range(len(train_op)):
            if train_op[i] == test[i][3]:
                traincorrect += 1

        print("DEPTH =",depth)
        print("ACCURACY | Train = ",(traincorrect/len(train_op))*100,"% | Test = ", (testcorrect/len(test_op))*100,'%')




        level = queue








if __name__ == "__main__":
    main()
