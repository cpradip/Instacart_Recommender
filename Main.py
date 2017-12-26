from data_processor import DataParser
from recommender import SVDpp, SVR
from evaluation.MeanPercentileRank import Evaluate

def ParseDataForProcessing():
	## Create intermediate data from raw data
	DataParser.ParseDataForIntermediateInfo()

	## Divide data into train and test
	DataParser.ParseDataIntoTrainTest()

	## Create test data with random products
	DataParser.CreateTestFileForRandomProducts()

def RecommendUsingMatrixFactorization():

	## Train and then recommend using SVDpp
	SVDpp.RecommendPredictions()

	## Evaluate the predicted results for test file
	#Evaluate("predictions_results_svdpp.csv")


def RecommendUsingRegression():
	## Parse training dataset with products info
	#DataParser.ParseDataWithProductsInfo("train_count_norm_1_10.csv")

	## Parse test dataset with products info
	#DataParser.ParseDataWithProductsInfo("test_count_norm_1_10.csv")

	## Train and then recommend using SVR
	SVR.RecommendPredictions()

	## Evaluate the predicted results for test file
	#Evaluate("predictions_results_svr.csv")



###-----------------------------------------------------------------###
	# Main file to start the recommendation procedure.
###-----------------------------------------------------------------###
if __name__ == "__main__":
	#ParseDataForProcessing()

	#RecommendUsingMatrixFactorization()
	
	RecommendUsingRegression()
