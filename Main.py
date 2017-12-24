from data_processor.DataParser import ParseDataIntoTrainTest, CreateTestFileForRandomProducts
from recommender.SVDpp import RecommendPredictions
from evaluation.MeanPercentileRank import Evaluate

###-----------------------------------------------------------------###
	# Main file to start the recommendation procedure.
###-----------------------------------------------------------------###
if __name__ == "__main__":
	## Divide data into train and test
	ParseDataIntoTrainTest()

	## Create test data with random products
	CreateTestFileForRandomProducts()

	## Recommend Predictions for the test files
	RecommendPredictions()

	## Evaluate the predicted results for test file
	Evaluate("predictions_results.csv")