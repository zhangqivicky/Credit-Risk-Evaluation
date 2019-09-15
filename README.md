# Home Credit Default Risk Prediction

## Introduction
Credit risk evaluation is one of the important topics in finance. The evaluation is highly challenging and one of the main reasons is that the records of credit histories are sometimes not sufficient. As a result, many people struggle to get loans. The reason why people want to study this dataset is that they hope to help this underserved population have a positive loan experience. The primary task is to develop some reliable models to predict clients' repayment abilities and credit risk based on a variety of alternative data provided by Home Credit.

"target" variable is the prediction which indicates whether the user is positive or negative in credit risk.

The main reason why Home Credit wants to run this competition is to find more underserved population and provide lending service. They have used various machine learning models to make the predictions, and now they want to improve the performance of prediction so that some clients are not rejected due to insufficient data on credit. This action is business-motivated.

## Method
In this project, I have tried three different methods. They include linear regression, logistic regression and decision tree. From testing results, logistic regression performs better (stable and relatively good score) than other two. This makes sense because logistical regression is often used for prediction of binary results. In this project, the output (i.e. target) is binary. Therefore theoretically logistical regression can fit into the scenario well. In contrast, it is expected that linear regression can not do a good job because their output is continuous. In addition, decision tree seems to fit into this scenario because its output can be binary. However, when I tried decision tree, I found that the calculation is very complex when there are hundreds of features. The processing is very time consuming. In order to avoid taking too long time to get model built, I have to limit the features in each layer as well as the height of the tree. The results are not good (unstable and relatively low score). Therefore, I adopt logistic regression as main prediction model for this project. After some optimisation including normalization, finally I can use the model to get a AUC of 0.709.

## Data Source
https://www.kaggle.com/c/home-credit-default-risk