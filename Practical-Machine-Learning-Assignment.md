---
title: "Activity Recognition Using Predictive Analytics"
author: "Arkaprabha Bhattacharya"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---

## Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now possible to collect a large amount of data about personal activity relatively inexpensively. The aim of this project is to predict the manner in which participants perform a barbell lift. The data comes from http://groupware.les.inf.puc-rio.br/har wherein 6 participants were asked to perform the same set of exercises correctly and incorrectly with accelerometers placed on the belt, forearm, arm, and dumbell.  

For the purpose of this project, the following steps would be followed:

1. Data Preprocessing
2. Exploratory Analysis
3. Prediction Model Selection
4. Predicting Test Set Output

## Data Preprocessing 

First, we load the training and testing set from the online sources and then split the training set further into training and test sets. 


```r
library(caret)

trainURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainURL))
testing <- read.csv(url(testURL))

label <- createDataPartition(training$classe, p = 0.7, list = FALSE)
train <- training[label, ]
test <- training[-label, ]
```

From among 160 variables present in the dataset, some variables have nearly zero variance whereas some contain a lot of NA terms which need to be excluded from the dataset. Moreover, other 5 variables used for identification can also be removed. 


```r
NZV <- nearZeroVar(train)
train <- train[ ,-NZV]
test <- test[ ,-NZV]

label <- apply(train, 2, function(x) mean(is.na(x))) > 0.95
train <- train[, -which(label, label == FALSE)]
test <- test[, -which(label, label == FALSE)]

train <- train[ , -(1:5)]
test <- test[ , -(1:5)]
```

As a result of the preprocessing steps, we were able to reduce 160 variables to 54.

## Exploratory Analysis

Now that we have cleaned the dataset off absolutely useless varibles, we shall look at the dependence of these variables on each other through a correlation plot. 


```r
library(corrplot)
```

```
## corrplot 0.92 loaded
```

```r
corrMat <- cor(train[,-54])
corrplot(corrMat, method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0,0,0))
```

![](Practical-Machine-Learning-Assignment_files/figure-html/CorrelationPlot-1.png)<!-- -->

In the plot above, darker gradient correspond to having high correlation. A Principal Component Analysis can be run to further reduce the correlated variables but we aren't doing that due to the number of correlations being quite few.

## Prediction Model Selection

We will use 3 methods to model the training set and thereby choose the one having the best accuracy to predict the outcome variable in the testing set. The methods are Decision Tree, Random Forest and Generalized Boosted Model.

A confusion matrix plotted at the end of each model will help visualize the analysis better.

### Decision Tree


```r
library(rpart)
library(rpart.plot)
library(rattle)
set.seed(13908)
modelDT <- rpart(classe ~ ., data = train, method = "class")
fancyRpartPlot(modelDT)
```

![](Practical-Machine-Learning-Assignment_files/figure-html/DecisionTree-1.png)<!-- -->

```r
predictDT <- predict(modelDT, test, type = "class")
confMatDT <- confusionMatrix(predictDT, as.factor(test$classe))
confMatDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1512  227   41   91   52
##          B   47  625   39   25   24
##          C   20   88  850  149   70
##          D   78  145   71  604  126
##          E   17   54   25   95  810
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7478          
##                  95% CI : (0.7365, 0.7589)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6798          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9032   0.5487   0.8285   0.6266   0.7486
## Specificity            0.9024   0.9716   0.9327   0.9147   0.9602
## Pos Pred Value         0.7863   0.8224   0.7222   0.5898   0.8092
## Neg Pred Value         0.9591   0.8997   0.9626   0.9259   0.9443
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2569   0.1062   0.1444   0.1026   0.1376
## Detection Prevalence   0.3268   0.1291   0.2000   0.1740   0.1701
## Balanced Accuracy      0.9028   0.7601   0.8806   0.7706   0.8544
```

### Random Forest


```r
library(caret)
set.seed(13908)
control <- trainControl(method = "cv", number = 3, verboseIter=FALSE)
modelRF <- train(classe ~ ., data = train, method = "rf", trControl = control)
modelRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.23%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    1    0    0    0 0.0002560164
## B    6 2648    4    0    0 0.0037622272
## C    0    5 2390    1    0 0.0025041736
## D    0    0    8 2243    1 0.0039964476
## E    0    1    0    5 2519 0.0023762376
```

```r
predictRF <- predict(modelRF, test)
confMatRF <- confusionMatrix(predictRF, as.factor(test$classe))
confMatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    0
##          B    0 1138    2    0    0
##          C    0    1 1024    2    0
##          D    0    0    0  962    0
##          E    1    0    0    0 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9978, 0.9996)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9991   0.9981   0.9979   1.0000
## Specificity            1.0000   0.9996   0.9994   1.0000   0.9998
## Pos Pred Value         1.0000   0.9982   0.9971   1.0000   0.9991
## Neg Pred Value         0.9998   0.9998   0.9996   0.9996   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1934   0.1740   0.1635   0.1839
## Detection Prevalence   0.2843   0.1937   0.1745   0.1635   0.1840
## Balanced Accuracy      0.9997   0.9994   0.9987   0.9990   0.9999
```

### Generalized Boosted Model


```r
library(caret)
set.seed(13908)
control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, verboseIter = FALSE)
modelGBM <- train(classe ~ ., data = train, trControl = control, method = "gbm", verbose = FALSE)
modelGBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 53 had non-zero influence.
```

```r
predictGBM <- predict(modelGBM, test)
confMatGBM <- confusionMatrix(predictGBM, as.factor(test$classe))
confMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    3    0    0    0
##          B    1 1122   10    4    4
##          C    0   13 1012    7    3
##          D    0    1    4  953    7
##          E    0    0    0    0 1068
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9903          
##                  95% CI : (0.9875, 0.9927)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9877          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9851   0.9864   0.9886   0.9871
## Specificity            0.9993   0.9960   0.9953   0.9976   1.0000
## Pos Pred Value         0.9982   0.9833   0.9778   0.9876   1.0000
## Neg Pred Value         0.9998   0.9964   0.9971   0.9978   0.9971
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1907   0.1720   0.1619   0.1815
## Detection Prevalence   0.2848   0.1939   0.1759   0.1640   0.1815
## Balanced Accuracy      0.9993   0.9905   0.9908   0.9931   0.9935
```

As Random Forest offers the maximum accuracy of 99.75%, we will go with Random Forest Model to predict our test data class variable.

## Predicting Test Set Output


```r
predictRF <- predict(modelRF, testing)
predictRF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
