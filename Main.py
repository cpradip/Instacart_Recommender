from data_processor.DataParser import ParseDataIntoTrainTest, CreateTestFileForRandomProducts, ParseDataWithProductsInfo
from recommender.SVDpp import RecommendPredictions
from evaluation.MeanPercentileRank import Evaluate

def RecommendUsingMatrixFactorization():
	## Divide data into train and test
	ParseDataIntoTrainTest()

	## Create test data with random products
	CreateTestFileForRandomProducts()

	## Recommend Predictions for the test files
	RecommendPredictions()

	## Evaluate the predicted results for test file
	Evaluate("predictions_results.csv")


def RecommendUsingRegression():
	ParseDataWithProductsInfo("train_count_norm_1_10.csv")

	ParseDataWithProductsInfo("test_count_norm_1_10.csv")

###-----------------------------------------------------------------###
	# Main file to start the recommendation procedure.
###-----------------------------------------------------------------###
if __name__ == "__main__":
	#RecommendUsingMatrixFactorization()
	
	RecommendUsingRegression()
