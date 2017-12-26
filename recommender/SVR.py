import pandas as pan
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.externals import joblib

###----------------------------------------------------------------###
	# Initiate training procedure and recommend predictions
	# using Support Vector Regression with kernel 'rbf'
	# with Input = user_id, product_id, aisle_id, department_id
	# and Output = counts (used as a rating parameter to describe
	# user's preference)
###----------------------------------------------------------------###
def RecommendPredictions():
	print "Load Training Data ...."
	trainDF = pan.read_csv("data_source/userproducts_train_count_norm_1_10.csv",header=None, dtype={2:np.float16})

	trainDataset = trainDF.as_matrix(columns=[0,1,3,4])
	trainOutput = trainDF.as_matrix(columns=[2])
	trainOutput = np.reshape(trainOutput,(trainOutput.shape[0],))
	trainOutput[np.isnan(trainOutput)] = 10.0

	print "Train Dataset ..."
	algo = SVR(max_iter = 100)
	algo.fit(trainDataset, trainOutput)

	joblib.dump(algo, 'data_source/svr_trained_model.pkl')

	print "Load Testing Data ..."
	testDF = pan.read_csv("data_source/userproducts_test_count_norm_1_10.csv",header=None, dtype={2:np.float16})
	
	testDataset = testDF.as_matrix(columns=[0,1,3,4])
	testActualOutput = testDF.as_matrix(columns=[2])
	testActualOutput[np.isnan(testActualOutput)] = 1.0

	#algo = joblib.load('data_source/svr_trained_model.pkl') 

	print "Start Predictions ..."
	testPredictedOutput = algo.predict(testDataset)
	testPredictedOutput = np.reshape(testPredictedOutput, (testPredictedOutput.shape[0],1))

	try:
		os.remove("data_source/predictions_results_svr.csv")
	except OSError:
		pass
	
	print "Saving Prediction results in File"
	result = np.append(testActualOutput, testPredictedOutput, axis = 1)
	result = np.append(testDataset[:,0:2], result, axis =1)
	np.savetxt("data_source/predictions_results_svr.csv", result, delimiter=',', fmt='%1.3f')