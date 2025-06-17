# Load necessary packages
library(xgboost)
library(neuralnet)
library(dplyr)
library(ggplot2)
library(caret)

# Load saved models
xgb_model <- xgb.load("xgb_model.model")
load("bpnn_model_fixed.RData")

# Load dataset
df <- read.csv("DATA SET.csv")

# ------------------- Feature Setup (same as training) -------------------
features <- c("DOC", "PL__SIZE", "Variant", "Pond_Type", "POND_SIZE_Ha", "Stocking_Density",
              "Temp", "Temp_PM", "pH", "PH_PM", "DO", "Alkalinity", "Ammonia", "Nitrite",
              "Nitrate", "Salinity", "FTF1", "FTF2", "FTF3", "FTF4", "Daily__feed", "Cum_Feed")

df <- df[complete.cases(df[, c(features, "Avg_Weight")]), ]
dummies <- dummyVars(Avg_Weight ~ ., data = df[, c(features, "Avg_Weight")])
encoded_data <- predict(dummies, newdata = df)
encoded_df <- data.frame(encoded_data)
xgb_input_cols <- colnames(encoded_df)

# Create baseline input (median values of encoded data)
baseline <- apply(encoded_df, 2, median)
baseline_input <- as.data.frame(t(baseline))

# ------------------- Setup for Simulation -------------------
# Training set min-max for BPNN scaling
train_mins <- apply(encoded_df, 2, min)
train_maxs <- apply(encoded_df, 2, max)

min_max_scale_row <- function(row, mins, maxs) {
  scaled <- (row - mins) / (maxs - mins)
  scaled[is.nan(scaled)] <- 0
  return(as.data.frame(t(scaled)))
}

# Define percentage changes
pct_changes <- c(-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15)
sim_results <- data.frame(
  Change = paste0(pct_changes * 100, "%"),
  Adjusted_Cum_Feed = round(baseline_input$Cum_Feed * (1 + pct_changes), 2),
  XGBoost_Pred = NA,
  BPNN_Pred = NA
)

# ------------------- Simulation Loop -------------------
for (i in seq_along(pct_changes)) {
  temp_input <- baseline_input
  temp_input$Cum_Feed <- baseline_input$Cum_Feed * (1 + pct_changes[i])
  
  # Predict with XGBoost
  temp_row <- encoded_df[1, ]
  temp_row[] <- baseline[xgb_input_cols]
  temp_row["Cum_Feed"] <- baseline["Cum_Feed"] * (1 + pct_changes[i])
  xgb_input <- as.matrix(temp_row)
  sim_results$XGBoost_Pred[i] <- predict(xgb_model, xgb_input)
  
  # Scale temp_input properly for BPNN
  bpnn_input <- min_max_scale_row(unlist(temp_input), train_mins, train_maxs)
  colnames(bpnn_input) <- colnames(encoded_df)
  
  # Predict with BPNN
  sim_results$BPNN_Pred[i] <- as.numeric(neuralnet::compute(bpnn_model, bpnn_input)$net.result)
}

# Print results
print(sim_results)

# Plot
ggplot(sim_results, aes(x = Adjusted_Cum_Feed)) +
  geom_line(aes(y = XGBoost_Pred, color = "XGBoost"), size = 1.2) +
  geom_line(aes(y = BPNN_Pred, color = "BPNN"), size = 1.2) +
  geom_point(aes(y = XGBoost_Pred, color = "XGBoost"), size = 3) +
  geom_point(aes(y = BPNN_Pred, color = "BPNN"), size = 3) +
  geom_text(aes(y = XGBoost_Pred, label = round(XGBoost_Pred, 1)), hjust = -0.1, size = 3, color = "darkred") +
  geom_text(aes(y = BPNN_Pred, label = round(BPNN_Pred, 1)), hjust = 1.2, size = 3, color = "darkblue") +
  labs(
    title = "Effect of Cum_Feed Change on Predicted Shrimp Weight",
    x = "Adjusted Cum_Feed (kg)",
    y = "Predicted Shrimp Weight (g)",
    color = "Model"
  ) +
  theme_minimal(base_size = 14)

ggsave("cum_feed_simulation_plot.pdf", width = 8, height = 5)

# Save simulation predictions
write.csv(sim_results, "cum_feed_simulation_predictions.csv", row.names = FALSE)

# ------------------- Real Model Evaluation Using Actual Data -------------------

# Define metric functions
calc_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  mape <- mean(abs((actual - predicted) / actual)) * 100
  r2 <- cor(actual, predicted)^2
  return(c(RMSE = rmse, MAE = mae, MAPE = mape, R2 = r2))
}

# Actual values
actuals <- df$Avg_Weight

# XGBoost predictions
xgb_preds <- predict(xgb_model, as.matrix(encoded_df))

# BPNN predictions
scaled_input <- sweep(encoded_df, 2, train_mins, "-")
scaled_input <- sweep(scaled_input, 2, train_maxs - train_mins, "/")
scaled_input[is.na(scaled_input)] <- 0
bpnn_preds <- as.numeric(neuralnet::compute(bpnn_model, scaled_input)$net.result)

# Calculate performance metrics
xgb_eval <- calc_metrics(actuals, xgb_preds)
bpnn_eval <- calc_metrics(actuals, bpnn_preds)

# Save real evaluation metrics
real_metrics <- rbind(
  data.frame(Model = "XGBoost_Real", t(xgb_eval)),
  data.frame(Model = "BPNN_Real", t(bpnn_eval))
)

print(real_metrics)
write.csv(real_metrics, "model_real_performance_metrics.csv", row.names = FALSE)
