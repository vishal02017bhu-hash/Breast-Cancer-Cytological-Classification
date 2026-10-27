library(tidyverse)
library(tidymodels)
library(janitor)
library(vip)
library(ggcorrplot)

tidymodels_prefer()

df <- read_csv("C:/Users/Reddy_vk/Desktop/r paper/wdbc.csv", show_col_types = FALSE) |>
  clean_names() |>
  select(-any_of(c("id", "x33"))) |>
  mutate(diagnosis = factor(diagnosis, levels = c("M", "B")))

set.seed(123)
split <- initial_split(df, prop = 0.8, strata = diagnosis)
train <- training(split)
test <- testing(split)
folds <- vfold_cv(train, v = 10, repeats = 3, strata = diagnosis)

rec <- recipe(diagnosis ~ ., data = train) |>
  step_zv(all_predictors()) |>
  step_corr(all_numeric_predictors(), threshold = 0.9) |>
  step_normalize(all_numeric_predictors())

lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet")

rf_spec <- rand_forest(trees = 500, mtry = tune(), min_n = tune()) |>
  set_engine("ranger", importance = "impurity") |>
  set_mode("classification")

svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) |>
  set_engine("kernlab") |>
  set_mode("classification")

xgb_spec <- boost_tree(trees = 500, tree_depth = tune(), learn_rate = tune()) |>
  set_engine("xgboost") |>
  set_mode("classification")

wf_set <- workflow_set(
  preproc = list(base = rec),
  models = list(lasso = lasso_spec, rf = rf_spec, svm = svm_spec, xgb = xgb_spec)
)

tuned <- wf_set |>
  workflow_map(
    "tune_grid",
    resamples = folds,
    grid = 15,
    metrics = yardstick::metric_set(yardstick::roc_auc, yardstick::accuracy, yardstick::sens, yardstick::spec),
    control = control_grid(save_pred = TRUE, save_workflow = TRUE),
    seed = 123,
    verbose = FALSE
  )

ranking <- rank_results(tuned, rank_metric = "roc_auc", select_best = TRUE)
best_id <- ranking |> slice(1) |> pull(wflow_id)

best_params <- extract_workflow_set_result(tuned, best_id) |>
  select_best(metric = "roc_auc")

final_wf <- extract_workflow(tuned, best_id) |>
  finalize_workflow(best_params)

final_fit <- last_fit(
  final_wf,
  split,
  metrics = yardstick::metric_set(yardstick::accuracy, yardstick::sens, yardstick::spec, yardstick::precision, yardstick::f_meas, yardstick::roc_auc)
)

test_metrics <- collect_metrics(final_fit)
test_preds <- collect_predictions(final_fit)

cols <- c(M = "#E45756", B = "#4C78A8")

# Figure 1: class distribution
df |>
  ggplot(aes(diagnosis, fill = diagnosis)) +
  geom_bar(width = 0.6) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.4) +
  scale_fill_manual(values = cols) +
  labs(x = "Diagnosis", y = "Count") +
  theme_classic() +
  theme(legend.position = "none")

# Figure 3: correlation matrix
df |> select(-diagnosis) |> cor() |> ggcorrplot(type = "lower", tl.cex = 6)

# Figure 4: PCA
pca <- df |> select(-diagnosis) |> prcomp(scale. = TRUE)
pv <- summary(pca)$importance[2, 1:2]

pca$x |>
  as_tibble() |>
  mutate(diagnosis = df$diagnosis) |>
  ggplot(aes(PC1, PC2, colour = diagnosis)) +
  geom_point(alpha = 0.7) +
  scale_colour_manual(values = cols) +
  labs(x = paste0("PC1 (", round(100 * pv[1], 1), "%)"),
       y = paste0("PC2 (", round(100 * pv[2], 1), "%)")) +
  theme_classic()

# Figure 5: cross-validated AUC and accuracy
autoplot(tuned, select_best = TRUE, metric = c("roc_auc", "accuracy"))

# Figure 6: confusion matrix
test_preds |> conf_mat(diagnosis, .pred_class) |> autoplot(type = "heatmap")

# Figure 7: ROC curve
test_preds |> roc_curve(diagnosis, .pred_M) |> autoplot()

# Figure 8: variable importance
extract_workflow(final_fit) |>
  extract_fit_parsnip() |>
  vip(num_features = 10, aesthetics = list(fill = "#4C78A8"))

ranking |>
  filter(.metric %in% c("roc_auc", "accuracy", "sens", "spec")) |>
  select(wflow_id, .metric, mean, std_err)

test_metrics