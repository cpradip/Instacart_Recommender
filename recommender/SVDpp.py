import pandas as pan
import csv
from surprise import Dataset, SVDpp, accuracy, Reader

###----------------------------------------------------------------###
	# Initiate training procedure and recommend predictions
###----------------------------------------------------------------###
def RecommendPredictions():
	## Load train and test data into Dataframes
	trainDF = pan.read_csv("data_source/train_count_norm_1_10.csv", header=None,dtype={2:np.float16})
	testDF = pan.read_csv("data_source/test_count_norm_1_10.csv", header=None,dtype={2:np.float16})

	reader = Reader(rating_scale=(1, 10))

	print "Load train set...."
	dataTrain = Dataset.load_from_df(trainDF, reader = reader)
	trainset = dataTrain.build_full_trainset()
	
	print "Initiate Training ....."
	algo = SVDpp(n_epochs = 1000,lr_all = 0.01, reg_all=0.02, verbose=True)
	algo.train(trainset)

	print " Load test set..."
	dataTest = Dataset.load_from_df(testDF, reader = reader)
	testset = dataTest.build_full_trainset().build_testset()

	print "Start predictions"
	predictions = algo.test(testset)

	## Print predictions results into csv file
	resultFile = open("data_source/predictions_results.csv","a")
	csv_writer = csv.writer(resultFile)
	csv_writer.writerow(["uid","iid","r_id","est"])

	for item in predictions:
		predictionTuple = (item.uid, item.iid, item.r_ui, item.est)
		csv_writer.writerow(predictionTuple)

	resultFile.close()

	#rmse = accuracy.rmse(predictions, verbose=True)