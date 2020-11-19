# Import the salary dataset
library(readr)
library(plyr)
library(ROCR)

salary_train <- read_csv("SalaryData_Train.csv")
salary_test <- read_csv("SalaryData_Test.csv")
salary <- rbind(salary_train, salary_test)

View(salary)
str(salary)
attach(salary)

salary$Salary=factor(ifelse(salary$Salary==">50K","1","0"))

#removing data without relative information
salary$education=NULL
salary$relationship=NULL
salary$capitalgain=NULL
salary$capitalloss=NULL

colnames(salary)
col=c("occupation","race","sex","native")
salary[col] <- lapply(salary[col], factor)

salary$age=factor(cut(age,breaks=4,labels = FALSE))
salary$hoursperweek=factor(cut(hoursperweek,breaks = 4,labels = FALSE))
str(salary)

#combining to reduce levels
salary$workclass <- factor(gsub('^Federal-gov', 'Government', salary$workclass))
salary$workclass <- factor(gsub('^Local-gov', 'Government', salary$workclass))
salary$workclass <- factor(gsub('^State-gov', 'Government', salary$workclass))
salary$workclass=factor(gsub('^Self-emp-inc','self-emp',salary$workclass))
salary$workclass=factor(gsub('^Self-emp-not-inc','self-emp',salary$workclass))
str(salary$workclass)

salary$maritalstatus=factor(gsub('^Married-AF-spouse','Married',salary$maritalstatus))
salary$maritalstatus=factor(gsub('^Married-civ-spouse','Married',salary$maritalstatus))
salary$maritalstatus=factor(gsub('^Married-spouse-absent','Married',salary$maritalstatus))
salary$maritalstatus=factor(gsub('^Divorced','Single',salary$maritalstatus))
salary$maritalstatus=factor(gsub('^Separated','Single',salary$maritalstatus))
salary$maritalstatus=factor(gsub('^Widowed','Single',salary$maritalstatus))
salary$maritalstatus=factor(gsub('^Never-married','Single',salary$maritalstatus))
str(salary$maritalstatus)

str(salary$occupation)
levels(salary$occupation)
salary$occupation=factor(gsub('^Prof-specialty','Proffesional',salary$occupation))
salary$occupation=factor(gsub('^Craft-repair','Workers',salary$occupation))
salary$occupation=factor(gsub('^Exec-managerial','Administration',salary$occupation))
salary$occupation=factor(gsub('^Adm-clerical','Administration',salary$occupation))
salary$occupation=factor(gsub('^Other-service','Services',salary$occupation))
salary$occupation=factor(gsub('^Armed-Forces','Services',salary$occupation))
salary$occupation=factor(gsub('^Farming-fishing','Workers',salary$occupation))
salary$occupation=factor(gsub('^Handlers-cleaners','Workers',salary$occupation))
salary$occupation=factor(gsub('^Machine-op-inspct','Workers',salary$occupation))
salary$occupation=factor(gsub('^Priv-house-serv','Services',salary$occupation))
salary$occupation=factor(gsub('^Protective-serv','Services',salary$occupation))
salary$occupation=factor(gsub('^Tech-support','Workers',salary$occupation))
salary$occupation=factor(gsub('^Transport-moving','Workers',salary$occupation))

summary(salary$occupation)
salary$native=NULL

summary(salary)

#Partitioning the data into training and testing:
id=sample(2,nrow(salary),prob = c(0.8,0.2),replace = TRUE)
train=salary[id==1,]
test=salary[id==2,]

#Generating the model:
library(e1071)
#install.packages("caret")
library(caret)
model=naiveBayes(train, train$Salary)
model_pred=predict(model,test)
confusionMatrix(test$Salary,model_pred)

library(gmodels)
CrossTable(model_pred, test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

test_acc = mean(model_pred == test$Salary)
test_acc

# On Training Data
train_pred <- predict(model, train)

train_acc = mean(train_pred == train$Salary)
train_acc