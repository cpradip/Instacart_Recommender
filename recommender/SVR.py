import pandas as pan
import numpy as np
from sklearn.svm import SVR

def RecommendPredictions():
	print "Load Training Data ...."
	trainDF = pan.read_csv("data_source/userproducts_train_count_norm_1_10.csv",header=None, dtype={2:np.float16})
	print trainDF.dtypes
	trainDataset = trainDF.as_matrix(columns=[0,1,3,4])
	trainOutput = trainDF.as_matrix(columns=[2])

	print trainDataset.shape
	print "------------------------------"
	print trainOutput.shape

	trainOutput = np.reshape(trainOutput,(trainOutput.shape[0],))
	print trainOutput.shape

	nanOutput = trainOutput[np.isnan(trainOutput)]
	print nanOutput.shape
	for n in nanOutput:
		print n

def test():
	print "Train Dataset ..."
	algo = SVR(max_iter = 10)
	algo.fit(trainDataset, trainOutput)


	print "Load Testing Data ..."
	testDF = pan.read_csv("data_source/userproducts_test_count_norm_1_10.csv",header=None, dtype={2:np.float16})
	testDataset = testDF.as_matrix(columns=[0,1,3,4])
	testActualOutput = testDF.as_matrix(columns=[2])

	print "Start Predictions ..."
	testPredictedOutput = algo.predict(testDataset)
	print testPredictedOutput.shape

