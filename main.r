#Importing dataset
train <- read.csv("train.csv")
test <- read.csv("test.csv")

#Exploring dataset
head(train)
head(test)

summary(train)
str(train)
summary(test)
str(test)

#Checking for missing values
library(Amelia)
missmap(train, main='Missing Map',col=c('blue','black'),legend=FALSE)
missmap(test, main="Mising Map", col=c('blue','black'),legend=FALSE)

# Imputing NA values
library(DMwR)
train <- centralImputation(train)
test <- centralImputation(test)

#Explanatory Data Analysis
#Uni-Variate Analysis

library(ggplot2)
ggplot(data=train,aes(x=department)) +geom_bar(color="black",fill="dodgerblue",width=0.7) + theme_minimal()

ggplot(train,aes(is_promoted)) + geom_bar(fill="dodgerblue",color="black",width=0.7) + theme_minimal()

ggplot(train,aes(avg_training_score)) + 
    geom_bar(color="black",fill="dodgerblue") + 
    theme_minimal()
	
ggplot(train,aes(length_of_service)) + geom_bar(color="black",fill="dodgerblue") + theme_minimal()

ggplot(train,aes(previous_year_rating)) + geom_bar(color="black",fill="dodgerblue") + theme_minimal()

ggplot(train,aes(age)) + geom_bar(color="black",fill="dodgerblue") + theme_minimal()

ggplot(train,aes(education)) + geom_bar(color="black",fill="dodgerblue") + theme_minimal()

table(train$KPIs_met..80.)

library(plotrix)
v <- c(35517,19291)
lbl <- c("Not met KPI > 80%","Met KPI > 80%")
pie3D(v,labels=lbl,main="Employee's KPI",col=c("#0066ff","#99c2ff"))

table(train$gender)

v <- c(16312,38496)
lbl <- c("Female","Male")
pie3D(v,labels=lbl,main="Gender Ratio",col=c("#ff99ff","#ff99c2"))

ggplot(train,aes(awards_won.)) + geom_bar(color="black",fill="dodgerblue",width=0.7) + theme_minimal()

#Bi-Variate Analysis
ggplot(data = train,aes(x=department,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer(palette="Dark2")

ggplot(data = train,aes(x=KPIs_met..80.,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer()+theme_minimal()

ggplot(data = train,aes(x=age,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer(palette=7)+theme_minimal()

ggplot(data = train,aes(x=avg_training_score,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer()+theme_minimal()

ggplot(data = train,aes(x=previous_year_rating,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer(palette=17)+theme_minimal()

ggplot(data = train,aes(x=length_of_service,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer(palette=12)+theme_minimal()

ggplot(data = train,aes(x=awards_won.,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer(palette=18)+theme_minimal()

ggplot(data = train,aes(x=gender,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer(palette=5)+theme_minimal()

ggplot(data = train,aes(x=recruitment_channel,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer(palette=11)+theme_minimal()

ggplot(data = train,aes(x=no_of_trainings,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer(palette=01)+theme_minimal()

ggplot(data = train,aes(x=region,fill=factor(is_promoted)))+geom_bar(position="fill",width=0.7)+scale_fill_brewer(palette=14)+coord_flip()+theme_minimal()

#Modelling
#Creating Dummy for Categorical Variables
library(dummies)

train <- dummy.data.frame(train,sep="_")
test <- dummy.data.frame(test,sep="_")

head(train)
head(test)
dim(train)
dim(test)

#Creating training and testing data set
set.seed(101)
library(caTools)
split <- sample.split(train$is_promoted, SplitRatio = 0.8)
training_set <- subset(train, split == TRUE)
test_set <- subset(train, split == FALSE)

head(training_set)
head(test_set)

#Decission tree
library(rpart)

trainDT <- training_set
trainDT[-61] <- scale(trainDT[-61])
testDT <- test_set
testDT[-61] <- scale(testDT[-61])

head(trainDT)
head(testDT)

classifierDT <- rpart(is_promoted ~ .,data=trainDT)
predDT = predict(classifierDT, newdata = testDT[-61])
predDT <- ifelse (predDT > 0.5,1,0)
head(predDT)

library(caret)
actual <- as.factor(testDT$is_promoted)
predict <- as.factor(predDT)
confusionMatrix(actual,predict)	#Accuracy=0.9256

#Logistic Regression
trainLR <- trainDT
testLR <- testDT
classifierLR <- glm(is_promoted ~ ., family=binomial, data=trainLR)

predLR = predict(classifierLR, type = 'response', newdata = testDT[-61])
predLR = ifelse(predLR > 0.5, 1, 0)

head(predLR)

actual <- as.factor(testLR$is_promoted)
predict <- as.factor(predLR)
confusionMatrix(actual,predict) #Accuracy=0.9342

#K-Nearest Neighbors
trainKN <- trainDT
testKN <- trainKN

library(class)
predKN = knn(train = trainKN[, -61],
             test = testKN[, -61],
             cl = trainKN[, 61],
             k = 5,
             prob = TRUE)
head(predKN)

actual <- as.factor(testKN$is_promoted)
predict <- predKN
confusionMatrix(actual,predict)	#Accuracy=0.926

#ANN
library(h2o)
h2o.init(nthreads = -1)

trainAN <- trainDT
testAN <- testDT

classifierAN = h2o.deeplearning(y = 'is_promoted',
                         training_frame = as.h2o(trainAN),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 100,
                         train_samples_per_iteration = -2)

predAN = h2o.predict(classifierAN, newdata = as.h2o(testAN[-61]))
predAN = ifelse(predAN > 0.5,1,0)
predAN = as.vector(predAN)
head(predAN)

actual <- as.factor(testAN$is_promoted)
predict <- as.factor(predAN)
confusionMatrix(actual,predict)	#Accuracy0.938

#XGBoost
X_feature <- names(training_set)
X_feature <- X_feature[-61]
X_target <- training_set$is_promoted
X_targetT <- test_set$is_promoted

library(xgboost)

Xgtrain <- xgb.DMatrix(data=as.matrix(training_set[,X_feature]), label=X_target, missing=NA)
Xgtest <- xgb.DMatrix(data =as.matrix(test_set[,X_feature]),label=X_targetT, missing=NA)
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = Xgtrain, nrounds = 1000, nfold = 5, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 50, maximize = F)
xgbcv

xgb1 <- xgb.train (params = params, 
                   data = Xgtrain, 
                   nrounds = 24, 
                   watchlist = list(val=Xgtest,train=Xgtrain), 
                       print.every.n = 10,
                       early.stop.round = 10, 
                       maximize = F , 
                       eval_metric = "error")
xgb1

predxg <- predict (xgb1,Xgtest)
predxg <- ifelse (predxg > 0.5,1,0)
head(predxg)

actual <- as.factor(X_targetT)
predict <- as.factor(predxg)
confusionMatrix(actual, predict)	#Accuracy=0.9435

mat <- xgb.importance (feature_names = colnames(X_feature),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20])

#XGBoost with hyperparamter
fact_col <- colnames(training_set)[sapply(training_set,is.character)]

for(i in fact_col) set(training_set,j=i,value = factor(training_set[[i]]))
for (i in fact_col) set(test_set,j=i,value = factor(test_set[[i]]))

colnames(training_set) <- make.names(colnames(training_set),unique = T)
colnames(test_set) <- make.names(colnames(test_set),unique = T)

library(mlr)
traintask <- makeClassifTask (data = training_set,target = "is_promoted")
testtask <- makeClassifTask (data = test_set,target = "is_promoted")

lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 10L)

library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)
mytune$ylrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

xgmodel <- train(learner = lrn_tune,task = traintask)
xgmodel

xgpred <- predict(xgmodel,testtask)
xgpred

library(caret)
confusionMatrix(xgpred$data$response,xgpred$data$truth)	#Accuracy:0.9442

#Predicting the results of test set
library(xgboost)
X_feature <- names(train)
X_feature <- X_feature[-61]
X_target <- train$is_promoted

Xgtrain <- xgb.DMatrix(data=as.matrix(train[,X_feature]), label=X_target, missing=NA)
Xgtest <- xgb.DMatrix(data =as.matrix(test[,X_feature]), missing=NA)

params <- list()
params$objective <- "binary:logistic"
params$eta <- 0.1
params$max_depth <- 8
params$subsample <- 0.952
params$colsample_bytree <- 0.931
params$min_child_weight <- 9.77
params$eval_metric <- "error"
params$booster <- "gbtree"

modelXG <- xgb.train(params <- params, Xgtrain, nrounds <- 100)
predXG <- predict(modelXG, Xgtest)
predXG = ifelse(predXG > 0.5, 1, 0)
head(predXG)

# Submission
submission <- data.frame(employee_id = test$employee_id,is_promoted=predXG)
head(submission,n=20)

write.csv(submission, file= 'submit.csv', row.names =F)