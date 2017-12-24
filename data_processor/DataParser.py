import pandas as pan
import numpy as np
import random
import math


###----------------------------------------------------------------###
	# Divide Into Train and Test with user_id, product_id and counts
	# using Orders and Order_Products for eval_set = 'prior'
###----------------------------------------------------------------###
def ParseDataIntoTrainTest():
	print "Loading Orders ...."
	orderDF = pan.read_csv('data_source/orders.csv')
	
	print "Filtering with Order Priors ...."
	priorOrderDF = orderDF.loc[orderDF['eval_set'] == 'prior'][['order_id','user_id']]
	priorOrderDF.to_csv("data_source/priorOrders.csv")
	
	print "Loading Order Products ...."
	orderProductsDF = pan.read_csv('data_source/order_products__prior.csv')

	print "Joining Users with Products ...."
	userProductsDF = priorOrderDF.join(orderProductsDF.set_index('order_id'), on='order_id')[['user_id','product_id']]

	print "Combining Product count with Users ...."
	uniqueUserProdutsDF = userProductsDF.groupby(['user_id','product_id']).size().reset_index(name='counts')
	uniqueUserProdutsDF.to_csv("data_source/uniqueUserProduts.csv")

	print "Selecting Unique Users  ...."
	uniqueUserDF = uniqueUserProdutsDF.groupby(['user_id']).size().reset_index(name='counts')
	uniqueUserDF.to_csv("data_source/uniqueUsers.csv")

	print "Division of Data Started ...."
	trainDF = pan.DataFrame(columns = ['user_id', 'product_id','counts'])
	testDF = pan.DataFrame(columns = ['user_id', 'product_id','counts'])

	## CSV files for storing training and testing set
	train_csv = open('data_source/train_count_norm_1_10.csv', 'a')
	test_csv = open('data_source/test_count_norm_1_10.csv', 'a')

	counter = 0
	
	for index, row in uniqueUserDF.iterrows():
		## Random division of each user data into train and test
		count = row['counts']
		randomNumbers = range(0,count)
		random.shuffle(randomNumbers)

		trainCount = int(math.floor((count*0.7)))
		trainList = randomNumbers[0:trainCount]
		testList = randomNumbers[trainCount:count]
		
		tempDF = uniqueUserProdutsDF.loc[uniqueUserProdutsDF['user_id'] == row['user_id']]

		## Feature Scaling to range between 1 and 100
		min = tempDF['counts'].min()
		max = tempDF['counts'].max()
		tempDF['counts'] = 1 + (tempDF['counts'] - min)*9/(max - min)

		tempTrainDF = tempDF.iloc[trainList]
		tempTestDF = tempDF.iloc[testList]

		## Appending train and test data to corresponding files
		tempTrainDF.to_csv(train_csv, header=False, index=False)
		tempTestDF.to_csv(test_csv, header=False, index=False)

		## Counter to track progress
		counter = counter + 1
		if(counter == 10000):
			print index
			counter = 0

	train_csv.close()
	test_csv.close()


###-----------------------------------------------------------------###
	# Creates test file for all users with 1000 random product_id 
	# that has not yet been purchased by the user
###-----------------------------------------------------------------###
def CreateTestFileForRandomProducts():
	## Gets user-product list that user has already purchased
	uniqueUserProdutsDF = pan.read_csv("data_source/uniqueUserProduts.csv")
	userList = uniqueUserProdutsDF["user_id"].unique()

	resultFile = open("data_source/test_results_random.csv","a")
	csv_writer = csv.writer(resultFile)
	csv_writer.writerow(["user_id","product_id","counts"])

	for user in userList:
		## Gets products for a user
		userProductForUser = uniqueUserProdutsDF.loc[uniqueUserProdutsDF["user_id"] == user]
		productIdList = userProductForUser["product_id"].values
		
		# range of product id (1, 49688)
		productIds = range(0,49689)
		productIdArray = np.asarray(productIds)
		productIdArray = np.delete(productIdArray, productIdList)
		productIdArray = np.delete(productIdArray, 0)
		random.shuffle(productIdArray)

		## Write into csv file 1000 random products user has not purchased
		for i in range(0,1000):
			testTuple = (user, productIdArray[i], 0)
			csv_writer.writerow(testTuple)

	resultFile.close()

###-----------------------------------------------------------------###
	# Converts data with user_id, product_id, counts into Sparse Matrix 
	# with user x product with counts as values
###-----------------------------------------------------------------###
def ParseDataAsSparseMatrix(filename):
	orderDF = pan.read_csv('data_source/' + filename)
	userDF = pan.read_csv("uniqueUsers.csv")

	counter = 0
	sparseData = open('data_source/sparse' + filename, 'a')

	for index, row in userDF.iterrows():
		count = row['counts']
		tempDF = orderDF.loc[orderDF['user_id'] == row['user_id']]
		productIds = tempDF['product_id'].values
		values = tempDF['counts'].values
		
		rowData = np.zeros((50000,), dtype=np.int)
		rowData[productIds] = values

		rowDataTrans = np.transpose(rowData.reshape(50000,1))

		np.savetxt(sparseData,rowDataTrans,fmt='%i', delimiter=',')

		counter = counter + 1
		if(counter == 10000):
			print index
			counter = 0

	sparseData.close()