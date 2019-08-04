---
title: "Practical Machine Learning - Data Science Specialization"
author: "335emily"
date: "`r Sys.Date()`"
output:
  pdf_document: default
  html_document: default
  fig_caption: yes
---

## Overview and Executive Summary

The goal of this assisngment is to predict how well 20 barbell lifts were performed, using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Participants were asked to perform movements correctly (in which case, the classe vairable is A) and incorrectly in 4 different ways (classe variables are B, C, D and E). 

Training and validation data were provided. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Loading Data

First, load the training and test sets from the online sources:
```{r setup}
rawTrainingData <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
rawValidationData <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

Then, set a seed for reproducibility:
```{r seed}
set.seed(1245)
```

Load the required libaries
```{r libraries, results='hide', message=FALSE}
library(caret)
library(corrplot)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(rattle)
```

## Exploratory Data Analysis

First, review the dimensions of the data to understand the size of the dataset.
```{r dim}
dim(rawTrainingData)
```

Then, look use the head functions to review some of the data and the column names. The results are excluded from the report to conserve space.
```{r head, results='hide'}
head(rawTrainingData)
```

## Cleaning Data

### Removing Variables with many NA or Blank Values
Looking at the results from the head function result, there are many NAs in the data. Therefore, it would be useful to understand how many rows are complete. 

```{r complete}
sum(complete.cases(rawTrainingData))
```

See that only `r sum(complete.cases(rawTrainingData))` rows of the `r nrow(rawTrainingData)` rows in the training data  are complete. Therefore, it is likely that many of the colummns will not be useful for prediction, as they contain NA values or missing observations that could create errors during training. These can be removed.

```{r NAcount}
maxNAPerVariable = 20
maxNACount <- nrow(rawTrainingData) / 100 * maxNAPerVariable
removeColumns <- which(colSums(is.na(rawTrainingData) | rawTrainingData=="") > maxNACount)
clean01TrainingData <- rawTrainingData[,-removeColumns]
clean01ValidationData <- rawValidationData[,-removeColumns]
```

By removing variables with more than `r maxNAPerVariable` NA or blank results, there are `r ncol(clean01TrainingData)` variables remaining.

### Removing Near Zero Variance Variables

Next, there are some variables with near zero variance, which can also be excluded.

```{r lowVarVarible}
lowVarVarible <-nearZeroVar(clean01TrainingData)
clean02TrainingData <- clean01TrainingData[,-lowVarVarible]
clean02ValidationData <- clean01ValidationData[,-lowVarVarible]
```

By further removing variables with low variance, there are `r ncol(clean02TrainingData)` variables remaining.

### Removing Identifiers and Timestamps

The final cleaning will remove variables used for identification or timestamps, assuming that neither will impact the `classe` variable,

```{r identifications}
cleanTrainingData <- clean02TrainingData[,-(1:5)]
cleanValidationData <- clean02ValidationData[,-(1:5)]
```

After all cleaning, there are `r ncol(cleanTrainingData)` variables variables remaining of the original `r ncol(rawTrainingData)` variables.

## Cross Validation
In order to perform cross validation on the features and the model, the training set will be split into a training set and a test set.

```{r split testing}
inTrain <- createDataPartition(y=cleanTrainingData$classe,p=0.75, list=FALSE)
training <- cleanTrainingData[inTrain,]
test <- cleanTrainingData[-inTrain,]
```

The `training` dataset will be used to create the model. The `test` dataset will be used for cross-validation of variable and model selection. Finally, the `cleanValidationData` dataset set will be used as a final validation set in the assignment.

## More Exploratory Data Analysis

### Correlation Review

Now that the data is clean and split, and the unhelpful variables are removed, the next step is to review if the variables are correlated to each other.

```{r correlation}
corrMat <- cor(cleanTrainingData[,-(ncol(cleanTrainingData))])
```

```{r figs, echo=FALSE, fig.width=7,fig.height=6,fig.cap="\\label{fig:figs}Fig 1 correlation"}
corrplot(corrMat, method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0,0,0))
```

In the plot, darker colours demonstrate a higher correlation. 

There are very few correlated variables, so a Principal Component Analysis is not productive, and therefore will not be performed.

## Model Selection

Four methods were considered for creating the prediction model: Linear Regression Model with Multiple Covariates, Decision Tree, Random Forest and a Generalized Boosted Model.

The model with the best accuracy found in during cross-validation will be selected for the final validation. Specifically, models will be built using the training data, and cross-validation will be performed using the test data. The model with the highest accuracy will be used to predict the 20 new test cases.

### 1. Linear Regression Model with Multiple Covariates

There is not sufficient information to determine which variables to include in the regression, therefore this model will not be tested. Including them all would lead to overfitting.

### 2. Decision Tree

```{r DecisionTree, message = FALSE, warning = FALSE}
modelDT <- rpart(classe ~ ., data = training, method = "class")
rpart.plot(modelDT)
predictDT <- predict(modelDT, test, type = "class")
confMatDT <- confusionMatrix(predictDT, test$classe)
accuracyDT <- confMatDT$overall['Accuracy']
```

### 3. Random Forest

```{r RandomForest, message = FALSE}
control <- trainControl(method = "cv", number = 3, verboseIter=FALSE)
modelRF <- train(classe ~ ., data = training, method = "rf", trControl = control)
modelRF$finalModel
predictRF <- predict(modelRF, test)
confMatRF <- confusionMatrix(predictRF, test$classe)
accuracyRF <- confMatRF$overall['Accuracy']
```

### 4. Generalized Boosted Model

```{r GBM, message = FALSE}
control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, verboseIter = FALSE)
modelGBM <- train(classe ~ ., data = training, trControl = control, method = "gbm", verbose = FALSE)
modelGBM$finalModel
predictGBM <- predict(modelGBM, test)
confMatGBM <- confusionMatrix(predictGBM, test$classe)
accuracyGBM <- confMatGBM$overall['Accuracy']
```

### Comparing Accuracy

Of the three models created, the accuracies are:
* Decision Tree: `r accuracyDT`
* Random Forest: `r accurarcyRF`
* Generalized Boosted Mode: `r accuracyGBM`

The Random Forest model has the highest accuracy, and will be used to predict the class value of the validation data

## Predicting Class Variable in the Validation Set

```{r TestSetPrediction, messages = FALSE}
predictRF <- predict(modelRF, cleanValidationData)
predictRF
```