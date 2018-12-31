# DataScienceChallenge
#Exploratory Data Analysis
#a) Relevant metrics
Total number of queries logged in the given dataset: 3451202
Total unique Queries: 1244496
Total unique URLS: 389403
Total number of users (unique): 66000
Average query per user: 52.2909393939394
Total number of queries with click events: 1783961
Total number of queries without click events: 1667241

Interesting Insights:
#The plot shows a dip in the number of queries on a day. Maintence activities on the search or downtime due to other issues could be the reason for the drop 
#The plot shows the highest number of searches occured on "Sundays" followed by "Monday". More ads on these days are recommended 
#The plot shows the highest number of searches occured in the "evening" followed by "night". More ads during evening time is beneficial 
#Sunday and Monday have the highest number of instances of user clicking on the url
Percentage of searches with spelling mistakes: 36.73111570983095
#EVening time has the highest number of instances of user clicking on the url
#lesser the number of words in the search higher, the instances of navigating to the url

#b) I would like to tell the following information to the Product manager of the search Engine
#Target customers with ads in the evening and night time
#Target customers with ads on either Sunday or Monday
#Many users who visit your search engine are predominantly searching for other search engine providers such as Google and Yahoo. Focus on building a competitive strategy to compete with industry leaders
#Ad a spell checker and correcter which would help users to obtain desired search results quickly. 36% of the queries contain spelling mistakes.
#Give page suggestions while typing so that users would avoid typing long sentences to search

#c) common queries
#The most common queries are Google, Ebay, Yahpp, myspace, and mapquest#d) 

#d) Queries leading to no click
#Only 14% of the queries that are misspelled had clicks
#Queries that has number of words higher than 10 and queries that have spelling mistakes
#Queries with Item Rank 0 has always led to a click

#e) Queries leading to click
#Queries that have  NumOfWords <5 always lead to click

#f) Which queries doen't seem to have relevant results
#Those queries that have higher rank don't seem to have relevant results

#h)common URL
#Websites such as google, apple are the commonly searched websites


#Clustering Analysis

#a) How will I cluster users?
#Potential respondants to advertisements vs others
#Based on the time of the day - Active hours - Students - evening/night, kids - Afternoon and evening,
#housewife - Afternoon and evening
#those who search with short query vs long query
#Based on the search content - Queries have to categorized into eg. kids search, teenager, adult


#b) Features considered
#I will consider the following features
#Time of the day, 
#number of words in a query
#Misspelled or not


#c) Clustering algorithm
#I will use kmeans clustering as it doesn't require our data to be distributed normally
#It is the most effective algorithm
#Easy to implement
#Works well with ordinal and nominal data too


#d)Choosing the right number of clusters
#I will use Elbow method to choose the optimum number of clusters
#Elbow method -> sum of square errors -vs number of clusters


#e) Distance metrics for clustering
#I will use Mahalanobis distance
#It takes covariance into consideration which lead tp elliptical decision boundaries
#Euclidean distance works well in a dataset which contains same units across all the input variables

#Classification
#Accuracy of the prediction model is 60.6%
#Sensitivity is good
#specificity is relatively low
#That is only 47% of the click not occuring condition has been captured out of the entire dataset.

#b)
#I have dereived features such as 
    #NumOfWords and is_missSpelled from "Query", 
    #weekday and TimeOfDayCorrected from "QueryTime"
#for the analysis
    

#c) I consider multiple factors while deciding on the right models to build. They are
    #Data type of the target variable
    #Linearity of the data
    #Data type of the input variables
    #Distribution of the target variable
    #balance of the data
	#Distribution of the error 
    
    
#d) I use k fold cross validation method with k value as 5.
    
    
#e) Metrics to support the reliability of the model:
    #R Squared
    #Accuracy of the model
    #Sensitivity
    #Specificity
    #Precision
    #ROC curve
    #p-value
