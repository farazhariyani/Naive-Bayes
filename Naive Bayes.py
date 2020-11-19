#import packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Loading the data set
salary_train = pd.read_csv("SalaryData_Train.csv",encoding = "ISO-8859-1")
salary_test = pd.read_csv("SalaryData_Test.csv",encoding = "ISO-8859-1")

#combining training and testing data to process the data
salary_data = salary_train.append(salary_test)

#removing irrelevant information
salary_data.drop(["education","relationship","capitalgain","capitalloss"], axis = 1, inplace = True)

#converting into binary
lb=LabelEncoder()
salary_data["age"]=lb.fit_transform(salary_data["age"])
salary_data["workclass"]=lb.fit_transform(salary_data["workclass"])
salary_data["maritalstatus"]=lb.fit_transform(salary_data["maritalstatus"])
salary_data["occupation"]=lb.fit_transform(salary_data["occupation"])
salary_data["race"]=lb.fit_transform(salary_data["race"])
salary_data["sex"]=lb.fit_transform(salary_data["sex"])
salary_data["hoursperweek"]=lb.fit_transform(salary_data["hoursperweek"])
salary_data["native"]=lb.fit_transform(salary_data["native"])
salary_data["Salary"]=lb.fit_transform(salary_data["Salary"])
salary_data.describe()
salary_data

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame
x = norm_func(salary_data.iloc[:, :9])
x.describe()

#target
y = salary_data.iloc[:, 9:]
y

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 9)

# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()

#for train data
classifier_mb.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = classifier_mb.predict(x_train) # store the prediction data
accuracy_score(y_train,y_pred) # calculate the accuracy

#for test data
classifier_mb.fit(x_test, y_test)
y_pred = classifier_mb.predict(x_test) 
accuracy_score(y_test,y_pred)