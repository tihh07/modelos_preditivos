#Packages

library(tidyverse)
library(tidymodels)

#Data
data(PimaIndiansDiabetes, package = "mlbench")

data_pima <- PimaIndiansDiabetes %>% janitor::clean_names()

data_pima_clean <- 
  data_pima %>% 
  mutate(across(c("pregnant":"mass"), ~ifelse(.x==0,NA,.x))) 

#Explore
data_pima_clean %>% 
  select(glucose, insulin, mass, diabetes) %>% 
  GGally::ggpairs(aes(color = diabetes, alpha =0.3))

#Split the data
split_pima <- initial_split(data_pima_clean, strata = diabetes)
train_pima <- training(split_pima)
test_pima <- testing(split_pima)

resample_pima <- vfold_cv(train_pima, strata = diabetes, v=5) #KFold

#Pre processing
rec_pima <- 
  recipe(diabetes~., data = train_pima) %>% 
  step_impute_knn(all_predictors())

#Model Specification Hiper parametros
mdl_spec_rpart_pima <- 
  decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("classification") %>% 
  set_args(cost_complexity= tune(),
           tree_depth = tune(),
           min_n = tune())

#Model Workflow
wkfl_rpart_pima <- 
  workflow() %>% 
  add_recipe(rec_pima) %>% 
  add_model(mdl_spec_rpart_pima)

#Hyperparameter tune
# grid spec      ---------------------------------------------------------------
grid_pima <- 
  grid_regular(cost_complexity(),
               tree_depth(),
               min_n(),
               levels = 3)

# tune grid      ---------------------------------------------------------------
doParallel::registerDoParallel()

set.seed(123)

tune_grid_rpart_pima <- 
  tune_grid(wkfl_rpart_pima,
            resamples = resample_pima,
            grid = grid_pima,
            metrics = metric_set(roc_auc,accuracy),
            control = control_grid(save_pred = TRUE))

best_grid_rpart_pima <- tune_grid_rpart_pima %>% select_best(metric= "roc_auc")

#final wkfl
final_wkfl_rpart_pima <- 
  finalize_workflow(wkfl_rpart_pima, best_grid_rpart_pima)

#Model Evaluating
mdl_eval_rpart_pima <- 
  final_wkfl_rpart_pima %>% 
  last_fit(split_pima)

mdl_eval_rpart_pima %>% 
  extract_fit_engine() %>% 
  rpart.plot::rpart.plot(cex = 0.7,
                         type = 3,
                         roundint = F)

#Model Fit
mdl_fit_rpart_pima <- 
  final_wkfl_rpart_pima %>% 
  fit(data_pima_clean)


#Make predictions
new_data_pima <- 
  tribble(~pregnant, ~glucose, ~pressure,~triceps,~insulin, ~mass, ~pedigree, ~age,
          2,87,68,34,77,38,0.41,25)

data_pima_clean$fit <- predict(mdl_fit_rpart_pima, new_data = data_pima_clean)

#Variable Importance
mdl_fit_rpart_pima %>% 
  extract_fit_engine() %>% 
  vip::vip()

