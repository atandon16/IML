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
library(data.table)
library(iml)
library(ggplot2)
library(ranger)

future::plan("multisession")

#Import dataset
diabetes <- read.csv("project/diabetes.csv")
#The dataset has 253680 observations and 22 variables

task = as_task_classif(diabetes, target = "Diabetes_binary")
set.seed(2026)

idx <- sample(nrow(diabetes), 5000)
diabetes_small <- diabetes[idx, ]

task <- as_task_classif(diabetes_small, target = "Diabetes_binary")
task$positive <- "1"

categorical_selector <- selector_type(c("factor", "ordered"))

make_encoded_learner <- function(learner) {
  as_learner(
    po("encode", method = "treatment",
       affect_columns = categorical_selector) %>>%
      learner
  )
}

#RF
rf_learner <- lrn(
  "classif.ranger",
  predict_type = "prob",
  num.trees = 200,
  num.threads = 4
)

rf_model <- make_encoded_learner(rf_learner)
rf_model$train(task)

X <- diabetes_small[, colnames(diabetes_small) != "Diabetes_binary"]
y <- diabetes_small$Diabetes_binary

predictor_rf <- Predictor$new(
  model = rf_model,
  data = X,
  y = y,
  type = "prob"
)

#PDP
pdp_bmi <- FeatureEffect$new(
  predictor_rf,
  feature = "BMI",
  method = "pdp",
  grid.size = 10
)

plot(pdp_bmi)

#PDP+ICE
ice_bmi <- FeatureEffect$new(
  predictor_rf,
  feature = "BMI",
  method = "pdp+ice",
  grid.size = 10
)

plot(ice_bmi)

#SHAP
x_interest <- X[1, , drop = FALSE]

shap <- Shapley$new(
  predictor_rf,
  x.interest = x_interest
)

plot(shap)
shap$results

#Gradient-based
gradient_expl <- FeatureImp$new(
  predictor_rf,
  loss = "ce",
  compare = "difference"
)

plot(gradient_expl)
gradient_expl$results

#CV
rr_rf <- resample(
  task,
  rf_model,
  rsmp("cv", folds = 5)
)

rr_rf$aggregate(msr("classif.ce"))