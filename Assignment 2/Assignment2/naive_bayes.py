##Name - Abhik Naskar
##Roll No - 17CS30001
##Assignment 2


import pandas as pd

def count(train,colname,label,target_value):
    value = (train[colname] == label) & (train[target] == target_value)
    return len(train[value])

def count_target(train):
	count_0 = train[target][train[target] == 0].count()
	count_1 = train[target][train[target] == 1].count()
	return count_0, count_1

def probability(train, label, features, count_one, count_zero):
	for col in features:
		Prob[0][col] = {}
		Prob[1][col] = {}
		for category in labels:
			count_cat_zero = count(train, col, category, 0)
			Prob[0][col][category] = (count_cat_zero + 1)/(count_zero + 5)
			count_cat_one = count(train, col, category, 1)
			Prob[1][col][category] = (count_cat_one + 1)/(count_one + 5)
	return Prob

def predict(test, features, prob_0, prob_1, Prediction):
	for row in range(len(test)):
		proby_0 = prob_0
		proby_1 = prob_1
		for col in features:
			proby_0 *= Prob[0][col][test[col].loc[row]]
			proby_1 *= Prob[1][col][test[col].loc[row]]

		if(proby_0 > proby_1):
			Prediction.append(0)
		else:
			Prediction.append(1)

	return Prediction

def accuracy(predictions, test_data):
    count = 0
    for x in range(len(test_data)):
        if test_data[x] == predictions[x]:
            count = count + 1
    return (count/(len(test_data)))*100.0

if __name__ == '__main__':
	train = pd.read_csv('/Users/abhiknaskar/Desktop/data2_19.csv')
	test = pd.read_csv('/Users/abhiknaskar/Desktop/test2_19.csv')

	test_X = test.iloc[:,1:]

	test_Y = test.iloc[:,0]

	target = 'D'

	labels = [1,2,3,4,5]

	Prediction = []

	Prob = {0:{},1:{}}

	features_train = train.columns[train.columns != target]
	features_test = test.columns[test.columns != target]
	classes = train[target].unique()

	##print(features)
	##print(classes)

	count_zero, count_one = count_target(train)

	total = len(train[target])
	##print(total)

	prob_0 = count_zero/total
	prob_1 = count_one/total

	##print("P(y = 1): ",prob_1)
	##print("P(y = 0): ",prob_0)

	Prob = probability(train, labels, features_train, count_one, count_zero)

	Predictions = predict(test_X, features_test, prob_0, prob_1, Prediction)

	print(Predictions)

	Accuracy = accuracy(Predictions, test_Y)

	print("Accuracy: %.2f"% Accuracy,"%")
