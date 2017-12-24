import pandas as pan

###------------------------------------------------------------------###
	# Evaluate Mean Percentile Rank (MPR) from the prediction results
	# Get mean of test values with percentile of predicted values
###------------------------------------------------------------------###

def Evaluate(filename):
	print "Loading Unique Users .... "
	uniqueUsersDF =  pan.read_csv("data_source/uniqueUsers.csv")

	print "Loading Prediction Results ..."
	predictionResultsDF = pan.read_csv("data_source/" + filename)

	rankValue = 0
	totalPredictedVal = predictionResultsDF["r_ui"].sum(axis=0)

	print "Calculating rank ..."

	for index, row in uniqueUsersDF.iterrows():
		userId = row["user_id"]

		## Get prediction results for each user
		userResultDF = predictionResultsDF.loc[predictionResultsDF["uid"] == row["user_id"]]
		resultCount = len(userResultDF.index)

		## Calculate PR for each user
		for index1, predictionRow in userResultDF.iterrows():
			predictionVal = predictionRow["est"]
			actualVal = predictionRow["r_ui"]

			greaterVal = len(userResultDF.loc[userResultDF["est"] > predictionVal].index)
			equalVal = len(userResultDF.loc[userResultDF["est"] == predictionVal].index)

			## Percentile for each user product
			percentile = (greaterVal + 0.5*equalVal)*100/resultCount

			## Percentile Rank from each user product added to Rank
			rankValue = rankValue + percentile * actualVal/totalPredictedVal

	print rankValue

