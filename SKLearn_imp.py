import numpy as np
import pandas as pd
import time
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import math
import statistics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split 
from random import seed
from random import randrange
import graphviz
import csv
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.feature_selection import SelectKBest,f_classif
from guppy import hpy

from memory_profiler import profile
#import sys
seed(1)
from sklearn.cluster import AgglomerativeClustering

import sys
#Read in file
raw_data = []
Datafile = open("forestfires.csv", "r")

while True:
	theline = Datafile.readline()
	if len(theline) == 0:
		break
	readData = theline.split(",")

	for pos in range(len(readData)):
		readData[pos] = (readData[pos]);
	raw_data.append(readData)
Datafile.close()

df=raw_data
#Convert to pandas dataframe
df_dataframe = pd.DataFrame(df,index=range(len(df)), columns=df[0])
#Remove the first row is it contains repeated columns
df_dataframe = df_dataframe.iloc[1:]

df1 = df_dataframe[['X', 'Y']]


#Removing anomalies
def remove_outliers(data_selected):
	#Define new variable df_remove_anom as a copy of data selected
	df_remove_anom = data_selected.copy()


	#Initialised number removed to be zero
	number_removed =0
	row = len(df_remove_anom)
	#Define number of columns
	col = len(df_remove_anom[0])



	for j in range(col):
		#Calculate standard deviation of the column iterating over
		st_dv = np.std(df_remove_anom[:,j])
		#calculate the mean of the column
		mean = statistics.mean(df_remove_anom[:,j])
		#median = np.median(df_remove_anom[:,j])

		#Iterate down the rows 
		for i in range(row):
			#Test if the value in the array lies above or below 3 standard deviation
			if(df_remove_anom[i][j]>((3*st_dv))+mean) or df_remove_anom[i][j]<(mean -(3*st_dv)):
				#If true replace with 'None'
				df_remove_anom[i][j] = None
				#Add 1 to number removed
				number_removed +=1
	#Return new array
		
	return df_remove_anom	



#Data scaling
def standardise_data(data_selected):
	#Creating a copy of the paramter passed through
	standard_data = data_selected.copy()
	

	#Re-define rows and column numbers
	#Define number of rows
	row = len(standard_data)
	#Define number of columns
	col = len(standard_data[0])

	for j in range(col):
		#Find the mean of each column
		#mean_list = standard_data[:j].mean()
		mean_list = statistics.mean(standard_data[j])

		#Finding the standard deviation of each column
		#standar_deviation = (sum(([item[j] for item in df]-mean_list)**2)/(len(df[1])))**0.5
		standar_deviation = np.std(standard_data[j])

		#print(standar_deviation)
		for i in range(row):
				
			#Re-define the values in standard data to be the standard deviation
			standard_data[i][j] = (standard_data[i][j]- mean_list)/standar_deviation
			if (standard_data[i][j]-mean_list)==0:
				standard_data[i][j] = 0

	#print(standard_data)
	return standard_data

def UnivariateSelection(df):
	#Define the row range of the intial unaltered array
	row = range(0, len(df))
	#Creating a pandas data frame of original data
	df_Us = pd.DataFrame(data=df,index= row) 
	#Define x to be all the columns in the array
	x = df_Us.iloc[:,0:len(df[0])]
	#Selecting target variable
	y  = df_Us.iloc[:,8] 
	#Define 5 best_features by performing f-test
	best_climate_features = SelectKBest(score_func=f_classif, k=5)
	#Printing the the best features
	print("Optimal number of columns are",best_climate_features)
	#fit = best_features.fit(x,y) #Fitting against the data
	fit = best_climate_features.fit(x,y)
	#Keeping columns of data of best fit
	cols = best_climate_features.get_support(indices=True)
	#Printing the columns of the best features
	print("The optimal column numbers for univariate selection are",cols)
	#Keep column names of new best scores
	features_df_new = x.iloc[:,cols]
	
	return features_df_new

def perform_PCA(cleaned_data):
	#Define number of components to use
	pca = PCA(n_components=2)
	#Fitting the transformation
	principal_component_analysis = pca.fit_transform(cleaned_data)
	principal_Df = pd.DataFrame(data = principal_component_analysis, columns = ['principal component 1', 'principal component 2'])
	#Print explained variance for each column
	print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

	return principal_Df
#Clustering alogirthm for getting the labels
def gaussian_model(data_selected):
	
	time_start = time.time()
	#Creating a pandas data frame containing the selected data
	row = range(len(data_selected))
	#Creating a pandas dataframe
	df_gmm = pd.DataFrame(data=data_selected,index= row)


	#Fitting my data to the model and predicting the labels for n components and 
	#definied a random state. Fitted to the panas dataframe
	gmm = GaussianMixture(n_components=7, random_state=1).fit(df_gmm)
	#Predicitng the labels of the data
	labels = gmm.predict(df_gmm) 


	probs = gmm.predict_proba(df_gmm)
	size = 50 * probs.max(1) ** 2  # square emphasizes differences

	#Finding the optimal number of clusters to use through "Elbow method" of BIC
	#Define a range of number of clusters to test over
	n_components = np.arange(1, 10)
	#Defining a model witted to the dataframe. Test over number of defined components
	models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(df_gmm) for n in n_components]
	
	#Plotting bic for 'models' over the number of componenets
	plt.plot(n_components, [m.bic(df_gmm) for m in models], label='BIC')
	#Plotting aic for 'models' over the number of componenets
	plt.plot(n_components, [m.aic(df_gmm) for m in models], label='AIC')
	plt.xlabel('n_components');
	plt.xticks([1,2,3,4,5,6,7,8,9,10])
	plt.legend(loc='best')
	plt.savefig("BIC.png")
	plt.close()
	#calculate time for function
	time_elapsed = (time.time() - time_start)
	print("Gaussian mixture model computation time is", time_elapsed)
	
	return labels



@profile(precision = 30)
def decision_tree_classifier(data, max_d, min_sample, min_leaf,size_test_step):
	data_selected = data
	time_start = time.time()
	h = hpy()

	#Calculating the len of the array to identify the number of rows
	row = range(len(data_selected[:,]))
	#Creating a pandas dataframe
	DecisionTree = pd.DataFrame(data=data_selected,index= row)
	#Defining x as all the values in the dataframe
	X = DecisionTree.iloc[:,].values
	#Defining y as the cluster labels from the required clustering method
	y = gaussian_model(data_selected)
	#y = predict(X)

	#Creating a test train split of the data and labels with a pretermined random state and 20% test size
	#Initial step 0.3
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = size_test_step, random_state=11)

	# Parameters
	n_classes = 7
	plot_colors = "ryb"
	plot_step = 0.02

	#set and fit decision tree
	clf_class = DecisionTreeClassifier(max_depth=max_d, min_samples_split = min_sample, min_samples_leaf=min_leaf, criterion = "entropy", random_state=1).fit(x_train, y_train)
	#Making a prediction on the test dataset
	pred = clf_class.predict(x_test)

	plt.figure(dpi=400, figsize=(10,6))
	plot_tree(clf_class, filled=True)
	plt.savefig("tree2.png", dpi =  400)
	plt.close()
	#target_names = ['class 0', 'class 1', 'class 2']
	target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
	class_report = classification_report(y_test, pred)

	print(class_report)
	#write_class(class_report)
	accuracy_value = accuracy_score(y_test, pred)

	#Making a prediction based on the trained dataset to test for overfitting.
	y_train_pred = clf_class.predict(x_train)
	accuracy_value_test_data = accuracy_score(y_train, y_train_pred)

	#Print test and train accuracy scores
	print("Accuracy test data", accuracy_value, "Accuacy train data", accuracy_value_test_data)
	#Calculate time for function
	time_elapsed_tree = (time.time() - time_start)
	size = h.heap()
	print("memory is ", size)
	print("Decision tree time is", time_elapsed_tree)
	return accuracy_value,accuracy_value_test_data, time_elapsed_tree
	
	#df = pd.DataFrame(class_report).transpose()
	#print(class_report)	
	#class_report.to_csv('df.csv')

def write_class(class_report):
	report_data = []
	lines = class_report.split('\n')
	for line in lines[2:-5]:
		print(line)
		row = {}
		row_data = line.split('      ')
		print(row_data)
		row['class'] = row_data[0]
		row['precision'] = float(row_data[1])
		row['recall'] = float(row_data[2])
		row['f1_score'] = float(row_data[3])
		row['support'] = float(row_data[4])
		report_data.append(row)
	print(report_data)
	dataframe = pd.DataFrame.from_dict(row)
	dataframe.to_csv('classification_report.csv', index = False)
	
def write_labels(x_test,labels):
	#Initialise data
	data = []
	#Set to list
	x_test = x_test.tolist()
	for i in range(len(x_test)):
		#Append labels to x list
		x_test[i].append(labels[i])
		#Append test data to array
		data.append(x_test[i])	
	np.savetxt('data.csv', data, delimiter=',')

#Definie function to save acccuarcy score list to text file
def score_value_list(score_list):
	np.savetxt('test_train_entropy_overfit.csv', score_list, delimiter=',')
 
 #Removing feature names
df = df[1:]
#Initialing array
df_temp = []
#Iterating over the length of the array
for i in range(len(df)):
	temp_col = []
	#Iterating over length of sub list
	for j in range(len(df[0])):
		#Try except to catch categorical data
		try:
			#set values as float
			df[i][j] = float(df[i][j])
			#Initialise value
			value = 0
			#Test for float
			if  isinstance(df[i][j],float):
				#Set value to array value
				value = df[i][j]
				#Append temp list
				temp_col.append(value)	
		except ValueError:
			continue
	#Append list to array
	df_temp.append(temp_col)

#Standardise the data
df_removed = standardise_data(df_temp)
df_removed = np.array(df_removed)

#Removing outliers
data_selected_remove = remove_outliers(df_removed)

#Removing na rows
cleaned_data=data_selected_remove[~np.isnan(data_selected_remove).any(axis=1)]

#Set up as a numpy array
data_labels_new = np.array(cleaned_data)

#Selection the best features
best_features = UnivariateSelection(data_labels_new)

#Perfroming PCA
data_PCA = perform_PCA(best_features)
#Setting the reduced data as anumpy array
data_PCA=np.array(data_PCA)

#Returning the labels
labels = gaussian_model(data_PCA)

#Writing the data and labels to  text file
write_labels(data_PCA, labels)
#value = decision_tree_classifier(data_PCA, 9, 3, 1, 0.3 )

#print(value)
#decision_tree_classifier(data_labels,max_d, min_sample, min_leaf, criterion = "gini")
'''
#Applying labels to PCA
PCA_labels = []
for i in range(len(data_PCA)):
	temp_row = np.append(data_PCA[i],labels[i])
	PCA_labels.append(temp_row)

PCA_labels = np.array(PCA_labels)
'''

decision_tree_classifier(data_PCA, 8, 2, 1, 0.3 )
'''
score_list =[]
#for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
for i in np.linspace(0.1,0.9,9):
#for i in range(1,20):
	
	#min_samples_split=i
	temp_list =[]
	value, value_test, time_elapsed = decision_tree_classifier(data_PCA, 8, 4, 2, i )
	temp_list.append(value)
	temp_list.append(value_test)
	temp_list.append(i)
	temp_list.append(time_elapsed)
	#temp_list.append(i)
	score_list.append(temp_list)

print(score_list)
score_value_list(score_list)
'''


