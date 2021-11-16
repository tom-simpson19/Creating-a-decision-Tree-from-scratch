import numpy as np
from memory_profiler import profile
import math

raw_data_test = []
Datafile = open("Raw_tree_test_data.csv", "r")

while True:
    theline = Datafile.readline()
    if len(theline) == 0:
        break
    readData = theline.split(",")

    readData[0] = float(readData[0])
    readData[1] = int(readData[1].split('.')[0])
    raw_data_test.append(readData)
Datafile.close()

df_test=raw_data_test





#######################################
'''
This code was written and tested using 
python 3.8.0.

The data file used is from the test
data and labels from 
'''
#######################################

##############################################
'''
Class DecisionTree focuses on calculating the
split criterions and finding the best split
point.
'''
##############################################


#Build decision tree
class DecisionTree:
    def __init__(self, data, algorithm):
        self.data = data
        #initialising left and right list for later
        self.left = [] 
        self.right =[]
        #Defining algoirthm using
        self.algorithm = algorithm


    def entropy(self,list_left, list_right, parent_list, D, Dleft, Dright):
        #Initialising entropy for left and right branch
        entropyDleft = 0
        entropyDright = 0
        
        #Creating loops to calculate entropy for each left and right branch rejecting 0
        #as zero values are not valid
        for i in range(len(list_left)):
            if list_left[i] != 0:
                entropyDleft +=  -list_left[i] * math.log(list_left[i],2)
        for i in range(len(list_right)):
            if list_right[i] != 0:
                entropyDright += -list_right[i] * math.log(list_right[i],2)
        
        #Calculate the entropy of the parent node
        entropyParent = 0
        #Iterate over the length of the parent list
        for i in range(len(parent_list)):
            #Evaluate if the index value is zero
            #Not possible to calculate zero values
            if parent_list[i] != 0:
                #Calculate entropy equation
                entropyParent += (parent_list[i] / D ) * math.log(parent_list[i]/D,2)

        #Calculate the weighted average of the left and right entropy branch
        weighted_average = (entropyDleft * Dleft/D) + (entropyDright * Dright/D)
        #Calculate the overall information gain using the weighted average and parent node
        Information_gain = entropyParent - weighted_average
        #Return the negative information gain
        return -Information_gain
  
    def gini_index(self,pleft,pright, D, Dleft, Dright):

        #Calcuate gini of left banch
        giniDleft = 1 - sum(pleft)
        #Calcualte gini of the right branch
        giniDright = 1 - sum(pright)
        #Calculate the overall gini value
        gini = Dleft/D*giniDleft + Dright/D*giniDright
        return gini


    def split(self, list_mid):
        #Initialise power for later use
        power = 0
        #Assign the power value based on algorithm choice for later
        if self.algorithm == "gini_index":
            #If algorithm is gini index power is assigned two
            power = 2
        else:
            #Entropy parameter uses power = 1
            power = 1    
        value_list = []
        #Calculating values for each midpoint
        for midpoint in list_mid:
            #### split data into left and right according to midpoint
            left=[]
            right=[]
            
            for i in range(len(self.data)):
                #Testing for x<midpoint if true append to left list
                if (self.data[i][0])<=(midpoint):
                    left.append(self.data[i])
                    #Testing for x>midpoint if true append to right
                else:
                    right.append(self.data[i])
                
            #Initialise max value left and length list for later
            max_value_left = 0
            length_list = 0
            
            #Find the max value of list based on max value of class for left branch 
            for i in range(len(left)):
                #Find first value of each sublist
                for j in range(len(left[0])):
                    #If new class value is greater than current assign it as current
                    if left[i][1] > max_value_left:
                        max_value_left = left[i][1]
            #Initialising max value for right split
            max_value_right = 0

            #Find the max value of list based on max value of class for right branch 
            for i in range(len(right)):
                #Find first value of each sublist(continous value)
                for j in range(len(right[0])):
                    #If new class value is greater than current assign it as current
                    if right [i][1] > max_value_left:
                        max_value_left = right[i][1]

            #If max value of left is greter than right the length of list is equal to left
            if max_value_left > max_value_right:
                length_list = max_value_left
            else:
                length_list = max_value_left

            #Create and empty list in the range of "length_list"
            left_count_list = [0]*(length_list +1 )
            for pair in left:
                #If the class value in  pair in range of list
                if pair[1] in range(max_value_left+1):
                    i = pair[1]
                    #Add one to the index the corresponds to the value of the class label
                    left_count_list[i] += 1
         
            #Reapeating the count for the right branch
            right_count_list = [0]*(length_list +1 )
            for pair in right :
                if pair[1] in range(max_value_left+1):
                    i = pair[1]
                    right_count_list[i] += 1

        
            #Calculating total D over all data
            D = len(self.data)
            #Calculating D of the left branch
            Dleft = len(left)
            #Calculating D of the right branch
            Dright = len(right)

            #Initialise pleft and pright lists
            pleft = [0] * (length_list + 1)
            pright = [0] * (length_list + 1)

            #Iterate over the range of the length of left_count_list
            for i in range(len(left_count_list)):
                #Only iterate over none zero values
                if left_count_list[i] != 0:
                    #Calculate pleft
                    pleft[i] = (left_count_list[i]/Dleft) ** power
                else:
                    #Pass if value if zero
                    pass

            #Iterate over right_count_list
            for i in range(len(right_count_list)):
                #Account for zero values
                if right_count_list[i] != 0:
                    pright[i] = (right_count_list[i]/Dright) ** power
                else:
                    pass

            #Initialise parent list 
            parent_list = [0] * (length_list + 1)
            #Initialise gini values for left and right branch
            giniDleft = 1 
            giniDright = 1

            #Iterate over left count lists
            for i in range(len(left_count_list)):
                #Calcuate parent list
                parent_list[i] = left_count_list[i] + right_count_list[i]
                #Caluclate gini values of left branch
                giniDleft = giniDleft - pleft[i]
                #Caluclate gini value of right branch
                giniDright = giniDright - pright[i]
        
            #Performs split algorithm basd on user input
            if self.algorithm == "gini_index":
                print("gini index return")
                value = self.gini_index(pleft, pright, D, Dleft, Dright)
            elif self.algorithm == "entropy":
                print("entropy calculation")
                value = self.entropy(pleft, pright, parent_list, D, Dleft, Dright)

            #### save value alongside split point [[mid/splitpoint][gini_index]]

            value_list.append([value, midpoint])

        return value_list



    def _findBestSplit(self):
        #This method finds the best split of a dataset
		#### data = [[continuous, label], [con, label], [con, label]]

		#### get list of midpoints
        list_mid = []
        values = []
        labels = set()
        labels2 = set()

		#Iterate through items in data
        for item in self.data:
            values.append(item[0]) #Append first index to values
            #Adding all different continous variables
            labels.add(item[0])
            #Add al possible different labels
            labels2.add(item[1])
        values.sort() #Sort through values so in order

        #If split cannot be calulated return false
        if len(labels)==1 or len(labels2)==1:
            #Return false if length of either is false
            return False
		 
		 #Calucating the midpoint using the index plus next index
        for i in range(len(values)-1):
            if not values[i] == values[i+1]:
                value = (values[i]+values[i+1])/2
                list_mid.append(value)	

        ginis = self.split(list_mid)
        print("the ginis are", ginis)

		### loop back to start for next midpoint
        giniValues = [gini[0] for gini in ginis]

        minGini = min(giniValues)
        giniIndex = giniValues.index(minGini)

        best_midpoint = ginis[giniIndex][1]
        #Initialisng final left and right splits
        final_left = []
        final_right = []

        for i in range(len(self.data)):
				#test if new point is less than the best midpoint
                if (self.data[i][0])<=(best_midpoint):
                    #Append to left branch
                    final_left.append(self.data[i])
                else:
                    #Append to right branch
                    final_right.append(self.data[i])

		#### return gini, split, left, right
        final_right=sorted(final_right, key=lambda x: x[0])
        final_left=sorted(final_left, key=lambda x: x[0])
        #Print the final split lists
        print(final_left)
        print(final_right)
        
        return minGini, best_midpoint, final_left, final_right


class DecisionTree2:
    def __init__(self, data, max_depth = 3, min_sample_split = 2, algorithm = "gini_index", parent = None):
        self.data = data
        #print(self.data)
        self.left = None #Initialise list for later
        self.right = None
        
        self.parent = parent
        self.splitpoint = None
        #Paramter initialisation
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.algorithm = algorithm

    def length(self, tree, arr=None): 
        if not self.parent == None:
            return self.parent.length(1) + 1
        else:
            return 1
        if arr == None:
            #Used to reset array 
            arr = [] 
        #Append value of node to list
        arr.append(tree)
        #Used to traverse down all left branches 
        if not tree.left == None:
            length(tree.left, arr) 
        #Used to traverse down all right branches
        if not tree.right == None:
            length(tree.right, arr)
        #Returns length of the array(number nodes)
        return len(arr) 
    @profile(precision=8)
    def run(self):
        node = DecisionTree(self.data, self.algorithm)
        #Test if length data(right, left branch) is greater than the min number of sample split
        if len(self.data) >= self.min_sample_split:
            print(self.data)
        
        
            try:
                minGini, best_midpoint, final_left, final_right = node._findBestSplit()
                self.splitpoint = best_midpoint
                #Return final_left, final right to run to fin best split
                self.left = DecisionTree2(final_left, max_depth, min_sample_split, algorithm, parent=self)
                print(self.left)
                self.right = DecisionTree2(final_right, max_depth, min_sample_split, algorithm, parent=self)
                print(self.right)
                #size_left = self.left.length(self.left)
                #size_right = self.right.length(self.right)
                current_depth = self.length(1)
                #Test for max depth
                size_left = current_depth
                size_right = current_depth
                #Contiously run if less than max depth
                if size_right < self.max_depth:
                    self.right.run()
                #Continously run if less than max depth
                if size_left < self.max_depth:
                    self.left.run()

            except TypeError:
                #Used to reset array
                self.left = None
                print(self.left)
                self.right = None
                print(self.right)
            
        #return branches
        return self.right, self.left

    def get_node_label(self):
        #Calculates label of each branch
        labels = [element[-1] for element in self.data]
        #Returns the most common node
        return max(set(labels), key=labels.count)


    def get_label(self, predicted_data):
        if not self.splitpoint == None:
            if predicted_data[0] < self.splitpoint:
                return self.left.get_label(predicted_data)
            else:
                return self.right.get_label(predicted_data)

        return self.get_node_label()
        



def write_labels(x_test):   
    np.savetxt('data_raw_tree_gini.csv', x_test, delimiter = ',')

'''
@profile(precision=8)
def run_profile():
    raw_data = []
    Datafile = open("Raw_tree.csv", "r")   
    while True:
        theline = Datafile.readline()
        if len(theline) == 0:
            break
        readData = theline.split(",")

        readData[0] = float(readData[0])
        readData[1] = int(readData[1].split('.')[0])
        raw_data.append(readData)
    Datafile.close()

    df=raw_data
    '''

test_data = [[23,0],[3,1],[20,3],[7,4], [4,4],[63,6],[21,2],[55,1],[23,0],[3,1],[20,3],[7,4], [4,4],[63,6],[21,2],[55,1],[23,0],[3,1],[20,3],[7,4], [4,4],[63,6],[21,8],[52,1],[93,0],[3,4],[20,5],[7,4], [4,4],[4,6],[21,2],[55,1],[43,4]]
accuracy_list = [] 
#############
'''
Input default values here
of the form 
DecisionTree2(data, max_depth, min_sample_split, algorithm, 0)
0 is the default parent node do not change.
Algorithm follows as:
"gini_index" or "entropy"

'''
#############
list_size = int(len(test_data))


print("Please enter a value for max_depth parameter") 
while True:
    try:
        max_depth = int(input())

    except ValueError:
        print("Max depth can not be a string or negative in value.")
        continue
    if max_depth >= 1 and max_depth <= list_size:
        break
    else:
        print("Max depth can not be less than one or the length of the dataset, please re-enter")


print("Please eneter a value for min sample split")
while True:
    try:
        min_sample_split = int(input())

    except ValueError:
        print("Min sample split cannot be a string, please re-enter")
        continue
    if min_sample_split >= 2 and min_sample_split <= list_size:
        break
    else:
        print("Cannot split a sample with less than two values or equal to the list size please re-enter")


print("Algorithm choice is either gini_index or entropy.")
print("Enter a value for the algorithm split choice") 
while True:
    try:
        algorithm = str(input())

    except ValueError:
        print("Algorithm cannot be a integer, please re-enter")
        continue
    if algorithm == "gini_index" or algorithm == "entropy":
        break
    else:
        print("Not a valid algorithm please re-enter")


'''
max_depth = 5
min_sample_split = 5
algorithm = "entropy"
'''
tree = DecisionTree2(df_test, max_depth, min_sample_split, algorithm)
tree.run()
number_right = 0
        
new_label_list =[]
     
for datapoint in df_test:
    temp_list = []
    new_label = tree.get_label(datapoint)
    new_label = int(float(new_label))
    if new_label == datapoint[-1]:
        print(True)
        number_right += 1
    else:
        print(False)
    datapoint.append(new_label)
    new_label_list.append(datapoint)
print(len(new_label_list))
accuracy = number_right/len(new_label_list)
accuracy_list.append(accuracy)
print("Accuracy of the algorithm is {} %".format(accuracy))

    #np.array(new_label_list)
print(new_label_list)
print(accuracy_list)
write_labels(new_label_list)

#run_profile()
