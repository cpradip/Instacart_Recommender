import pandas as pan
import csv
import os
from surprise import Dataset, SVDpp, accuracy, Reader

###----------------------------------------------------------------###
	# Initiate training procedure and recommend predictions
	# using SVDpp with Input = user_id and product_id
	# and Output = counts (used as a rating parameter to describe
	# user's preference)
###----------------------------------------------------------------###
def RecommendPredictions():
	## Load train and test data into Dataframes
	trainDF = pan.read_csv("data_source/train_count_norm_1_10.csv", header=None,dtype={2:np.float16})
	trainDF = trainDF.fillna(10.0)

	reader = Reader(rating_scale=(1, 10))

	print "Load train set...."
	dataTrain = Dataset.load_from_df(trainDF[[0,1,2]], reader = reader)
	trainset = dataTrain.build_full_trainset()
	
	print "Initiate Training ....."
	algo = SVDpp(n_epochs = 10,lr_all = 0.01, reg_all=0.02, verbose=True)
	algo.train(trainset)

	## Predictions for test set with ground truth present
	print " Load test set..."
	testDF = pan.read_csv("data_source/test_count_norm_1_10.csv", header=None,dtype={2:np.float16})
	testDF = testDF.fillna(10.0)
	dataTest = Dataset.load_from_df(testDF[[0,1,2]], reader = reader)
	testset = dataTest.build_full_trainset().build_testset()

	print "Start predictions"
	predictions = algo.test(testset)

	try:
		os.remove("data_source/predictions_results_svdpp.csv")
	except OSError:
		pass

	print "Saving Prediction results in File"
	resultFile = open("data_source/predictions_results_svdpp.csv","a")
	csv_writer = csv.writer(resultFile)

	for item in predictions:
		predictionTuple = (item.uid, item.iid, item.r_ui, item.est)
		csv_writer.writerow(predictionTuple)

	resultFile.close()

	## Predictions for test set with random products present
	##	LEFT

	#rmse = accuracy.rmse(predictions, verbose=True)