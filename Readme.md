##-------------------------- Requirements -----------------------##

Instacart recommender: Python based recommendation system

Python Libraries used:
------------------------------------------------------------------

Numpy,
Pandas,
Scikit,
Sklearn,
Surprise,

Dataset:
------------------------------------------------------------------
1. Create a data_source folder along with other folders.
2. Insert 
   orders.csv, order_products__prior.csv and products.csv 
   files inside the data_source folder

##-------------------------- Main.py -----------------------##

Main file:

"python Main.py"
runs the main file

1. Intial step: It runs the data parser function (ParseDataForProcessing)
   which generate necessary files for recommendation process
2. Two types of recommendation algorithm used : 
   SVDpp (RecommendUsingMatrixFactorization)
   and 
   SVR (RecommendUsingRegression)

3. Uncomment required recommendation function from Main.py to run
   the function