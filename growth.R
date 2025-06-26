# Load necessary packages
library(xgboost)
library(dplyr)
library(ggplot2)
library(caret)
library(tidyr)
library(scales)

# Load saved model
xgb_model <- xgb.load("xgb_model.model")

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
xgb_input_cols <- colnames(encoded_df)  # These are the encoded numeric columns

# Create baseline input (median values of encoded data)
baseline <- apply(encoded_df, 2, median)
baseline_input <- as.data.frame(t(baseline))

# ------------------- Setup for Simulation -------------------
pct_changes <- seq(-0.5, 0.5, by = 0.1)
sim_results <- data.frame(
  Change = paste0(pct_changes * 100, "%"),
  Adjusted_Cum_Feed = round(baseline_input$Cum_Feed * (1 + pct_changes), 2),
  XGBoost_Pred = NA
)

for (i in seq_along(pct_changes)) {
  # Create full feature vector based on encoded_df structure
  temp_row <- encoded_df[1, ]
  temp_row[] <- baseline[xgb_input_cols]
  temp_row["Cum_Feed"] <- baseline["Cum_Feed"] * (1 + pct_changes[i])
  
  # XGBoost Prediction: Use only numeric encoded columns
  xgb_input <- as.matrix(temp_row[, xgb_input_cols])  # Select only encoded features
  sim_results$XGBoost_Pred[i] <- predict(xgb_model, xgb_input)
  
  # Debug: Check prediction
  print(paste("Simulation", i, "Change:", pct_changes[i],
              "XGBoost Prediction:", sim_results$XGBoost_Pred[i]))
}

# Print results
print(sim_results)

# Plot
ggplot(sim_results, aes(x = Adjusted_Cum_Feed, y = XGBoost_Pred)) +
  geom_line(color = "steelblue", size = 1.2) +
  geom_point(color = "steelblue", size = 3) +
  geom_text(aes(label = round(XGBoost_Pred, 1)), hjust = -0.1, size = 3, color = "darkred") +
  labs(
    title = "Effect of Cum_Feed Change on Predicted Shrimp Weight",
    x = "Adjusted Cum_Feed (kg)",
    y = "Predicted Shrimp Weight (g)"
  ) +
  theme_minimal(base_size = 14)

ggsave("cum_feed_simulation_plot.pdf", width = 8, height = 5)

# Save simulation predictions
write.csv(sim_results, "cum_feed_simulation_predictions.csv", row.names = FALSE)

#___________________________________________________________________________________

# ------------------- Real Model Evaluation Using Actual Data -------------------
calc_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  mape <- mean(abs((actual - predicted) / actual)) * 100
  r2 <- cor(actual, predicted)^2
  return(c(RMSE = rmse, MAE = mae, MAPE = mape, R2 = r2))
}

actuals <- df$Avg_Weight

# XGBoost predictions
xgb_preds <- predict(xgb_model, as.matrix(encoded_df))  # Already encoded, should work

# Calculate performance metrics
xgb_eval <- calc_metrics(actuals, xgb_preds)

# Save real evaluation metrics
real_metrics <- data.frame(Model = "XGBoost_Real", t(xgb_eval))

print(real_metrics)
write.csv(real_metrics, "model_real_performance_metrics.csv", row.names = FALSE)

# 2.1 DENSITY GRID (based on real range) -------------------
dens_seq  <- seq(6, 108, by = 4)            # PL m⁻²
variants  <- c("SIS_GROWTH_LINE", "SyAqua","SIS HARDY LINE")

grid2 <- expand_grid(Stocking_Density = dens_seq,
                     Variant = variants)

# Convert baseline to a data frame and replicate for each row
baseline_df <- as.data.frame(t(baseline))  # Transpose to make it a 1-row data frame
# Select only features excluding Stocking_Density and Variant to avoid conflicts
baseline_features <- setdiff(colnames(baseline_df), c("Stocking_Density", "Variant"))
grid2 <- grid2 %>%
  bind_cols(baseline_df[rep(1, nrow(grid2)), baseline_features]) %>%  # Bind only relevant baseline columns
  mutate(Stocking_Density = Stocking_Density,  # Retain from expand_grid
         Variant = Variant,                    # Retain from expand_grid
         Avg_Weight = baseline["Avg_Weight"],   # Add placeholder for Avg_Weight
         Pond_Type = sample(df$Pond_Type, size = nrow(grid2), replace = TRUE))  # Sample Pond_Type levels

# 2.2 PREDICT AVERAGE WEIGHT -------------------------------
# Encode Variant and Stocking_Density to match training data
grid2_encoded <- predict(dummies, newdata = grid2)
grid2$Pred_AW <- predict(xgb_model, as.matrix(grid2_encoded))

# 2.3 CALCULATE YIELD ONLY (kg/ha) -------------------------
grid2 <- grid2 %>%
  mutate(Yield_kg_ha = Pred_AW * Stocking_Density * 10000 / 1000)  # PL m⁻² × weight (g)

# 2.4 PLOT YIELD CURVES ------------------------------------
density_plot <- ggplot(grid2, aes(Stocking_Density, Yield_kg_ha, color = Variant)) +
  geom_line(linewidth = 1.1) +
  geom_point() +
  scale_x_continuous(breaks = pretty_breaks()) +
  scale_y_continuous(labels = comma_format()) +
  labs(title = "Predicted Yield vs Stocking Density by Genetic Line",
       x = "Stocking Density (PL m^-2)",  # Fixed label
       y = "Predicted Yield (kg ha^-1)",  # Fixed label
       color = "Genetic Line") +
  theme_minimal()

print(density_plot)  # Display in interactive session

# Save density plot
ggsave("density_simulation_plot.pdf", plot = density_plot, width = 8, height = 5)


# 2.5 PEAK YIELD SUMMARY -----------------------------------
peaks <- grid2 %>%
  group_by(Variant) %>%
  slice_max(Yield_kg_ha, n = 1, with_ties = FALSE) %>%
  select(Variant, Opt_Density = Stocking_Density,
         Pred_Avg_Weight = Pred_AW, Yield_kg_ha)
print(peaks)

