import pandas as pan
import numpy as np
from sklearn.svm import SVR

def RecommendPredictions():
	print "Load Training Data ...."
	trainDF = pan.read_csv("data_source/userproducts_train_count_norm_1_10.csv",header=None, dtype={2:np.float16})
	print trainDF.dtypes
	trainDataset = trainDF.as_matrix(columns=[0,1,3,4])
	trainOutput = trainDF.as_matrix(columns=[2])

	trainOutput = np.reshape(trainOutput,(trainOutput.shape[0],))
	print trainOutput.shape

	trainOutput[np.isnan(trainOutput)] = 10.0

	print "Train Dataset ..."
	algo = SVR(max_iter = 10)
	algo.fit(trainDataset, trainOutput)


	print "Load Testing Data ..."
	testDF = pan.read_csv("data_source/userproducts_test_count_norm_1_10.csv",header=None, dtype={2:np.float16})
	testDataset = testDF.as_matrix(columns=[0,1,3,4])
	testActualOutput = testDF.as_matrix(columns=[2])
	testActualOutput[np.isnan(testActualOutput)] = 1.0
	testActualOutput = np.reshape(testActualOutput,(testActualOutput.shape[0],))
	print testActualOutput.shape

	print "Start Predictions ..."
	testPredictedOutput = algo.predict(testDataset)
	print testActualOutput.shape
	print testPredictedOutput.shape

	result = np.append(testActualOutput, testPredictedOutput, axis = 1)
	np.savetxt("data_source/predictions_results_svr.csv", result, delimiter=',')