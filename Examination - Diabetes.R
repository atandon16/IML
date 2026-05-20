#### Option 1: CDC Diabetes Health Indicators
#For the CDC Diabetes Health Indicators dataset, you should build classification models predicting Diabetes_binary.


library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3mbo)
library(mlr3pipelines)
library(paradox)
library(future)
library(tidyverse)

future::plan("multisession") 
# future::plan("multicore")  # only works on Linux/macOS

#Import dataset
diabetes <- read.csv("C:/Users/THOL0099/Desktop/Diabetes.csv")
#The dataset has 253680 observations and 22 variables


#### 1. Pre-processing. Do you need to perform any pre-processing?

#I use table to check if all values match the variable description, e.g. binary variables being 0 or 1, and see missing values.
table(diabetes$Diabetes_binary, useNA = "ifany")
table(diabetes$HighBP, useNA = "ifany")
table(diabetes$HighChol, useNA = "ifany")
table(diabetes$CholCheck, useNA = "ifany")
table(diabetes$BMI, useNA = "ifany")
table(diabetes$Smoker, useNA = "ifany")
table(diabetes$Stroke, useNA = "ifany")
table(diabetes$HeartDiseaseorAttack, useNA = "ifany")
table(diabetes$PhysActivity, useNA = "ifany")
table(diabetes$Fruits, useNA = "ifany")
table(diabetes$Veggies, useNA = "ifany")
table(diabetes$HvyAlcoholConsump, useNA = "ifany")
table(diabetes$AnyHealthcare, useNA = "ifany")
table(diabetes$NoDocbcCost, useNA = "ifany")
table(diabetes$GenHlth, useNA = "ifany")
table(diabetes$MentHlth, useNA = "ifany")
table(diabetes$PhysHlth, useNA = "ifany")
table(diabetes$DiffWalk, useNA = "ifany")
table(diabetes$Sex, useNA = "ifany")
table(diabetes$Age, useNA = "ifany")
table(diabetes$Education, useNA = "ifany")
table(diabetes$Income, useNA = "ifany")

#All values are within variable range described on https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators 
#No missing values

sum(duplicated(diabetes))
24206/253680
#24206 duplicated values (9.5% of all observations)
#We chose not to delete duplicated rows, as we believe that more than one person with identical values for the 22 variables with few categories may be possible.


#### 2. Choice of models
#Which models are relevant for this task? Which models did you try, which model or models did you choose to focus on, and why?

#We start by define task in the dataset and splitting it (80% training 20% test)
task = as_task_classif(diabetes, target = "Diabetes_binary")
set.seed(2026)
splits = partition(task, ratio = 0.8)

task_train = task$clone()$filter(splits$train)
task_test = task$clone()$filter(splits$test)


#Logistic Regression. Logistic regression can model probabilities for a binary outcome (diabetes yes/no) 

my_lr_learner = lrn("classif.log_reg")

graph_learner_lr = as_learner(
  po("encode", method = "treatment", id = "binary_enc") %>>%
    po("scale") %>>%
    my_lr_learner
)

graph_learner_lr$train(task_train)
graph_learner_lr$predict(task_test)$score(msr("classif.ce"))



#Baseline with no features

baseline <- lrn("classif.featureless")$train(task_train)
baseline$predict(task_test)$score(msr("classif.ce"))




#Elastic net. Elastic net combines Lasso (L1) and Rigde (L2) regularization in a model
#First we create a learner.
my_elasticnet_learner = lrn(
  "classif.glmnet", 
  standardize = FALSE,
  alpha = to_tune(0, 1),
  lambda = to_tune(p_dbl(log(1e-5), log(1e1), trafo = exp))
  # Alternatively lambda = to_tune(1e-5, 1e1, logscale=TRUE)
)

#We create a graph that converts factor variables into dummy variables, standardizes predictors, and afterwards applies elastic net
graph_learner_elastic_net = as_learner(
  po("encode", method = "treatment", id = "binary_enc") %>>%
    po("scale") %>>%
    my_elasticnet_learner
)


#Now we apply 5-fold cross-validation (train on 4 folds, validate on 1 fold, repeat 5 times). 
#Cross-validation is used to reduce overfitting.
set.seed(2026)
instance = tune(
  tuner = tnr("random_search", batch_size=20), 
  task = task_train,
  learner = graph_learner_elastic_net,
  resampling = rsmp("cv", folds = 5), 
  measures = msr("classif.ce"),
  terminator = trm("evals", n_evals = 50) 
)

as.data.table(instance$archive$data)
instance$result_learner_param_vals

instance$result_y

# keep an untuned copy of the same graph for the auto_tuner example below
graph_learner_elastic_net_auto = graph_learner_elastic_net$clone(deep = TRUE)

# result on test set
graph_learner_elastic_net$param_set$values = instance$result_learner_param_vals
graph_learner_elastic_net$train(task_train)
graph_learner_elastic_net$predict(task_test)$score(msr("classif.ce"))


#### 3. Model performance
#How well do your models predict the desired outcome? Which performance metrics do you use, and why are they appropriate for your chosen problem?

#We used classification error (Number of incorrect predictions / Total number of predictions). 

#Logistic Regression
#Classification error (CE) = 0.1363923

#Baseline with no features
#Classification error (CE) = 0.139881

##Elastic net regularization (Best prediction)
#Classification error (CE) = 0.1357222 




#### 4. Interpretability and feature importance
#Which features, or combinations of features, appear important for prediction? How do they impact the prediction? 
#To what extent is it possible to understand or interpret the predictions made by your chosen model?

#Classification error: How far is the predictions of the model from the true values.
#Specificity: Ability to correctly identify true negatives. How many of the no-diabetes cases we predict truly dont have diabetes.
#Sensitivity: The ability to correctly identify true positives. How many of the diabetes cases we predict truly have diabetes.
#Accuracy: Overall correct predictions
#Precision


#### 5.Interpretability vs. accuracy
#You should explicitly discuss the trade-off between interpretability and predictive accuracy.

#The simpler model (logistic reg) is more interpretable but less accurate than e.g. elastic net regularization.
#Show examples with model naive model, logistic etc.


#### 6. Sensitive or protected features
#You should explicitly consider the role of potentially sensitive features in your chosen dataset.
#For the CDC Diabetes Health Indicators dataset, this this could for example be a variables such as BMI.
#In addition to your main final model, you should report a second model that does not make use of chosen sensitive variables, neither directly nor indirectly. 
#You should explain how you identified and handled such variables, and you should compare the original model and the restricted model. In particular, you should discuss what changed.


#The most relevant variable are sensitive
#Vizualisation for sensitive variables and non-sensitive for comparison

#Sensitive variables can be identified by allready established correlations e.g. increased risk of diabetes with increasing BMI and also by modelling the data.
#compare prediction of models without sensitive variables. We would expect less accuracy using the same models without the the sensitive variables.
