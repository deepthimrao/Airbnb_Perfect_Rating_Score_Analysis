
#load libraries
library(tidyverse)
library(caret)
library(tree)
library(class)
library(dplyr)
library(pROC)
library(rpart)
library(Metrics)
library(mlr)
library(ggplot2)
library(plotly)
library(randomForest)
library(glmnet)
library(tidytext)
library(ranger)
library(viridis)
library(ROCR)

set.seed(1)

#load data files
train_x <- read_csv("airbnb_train_x_2023.csv")
train_y <- read_csv("airbnb_train_y_2023.csv")
test_x <- read_csv("airbnb_test_x_2023.csv")

#Combine train and test data for cleaning, feature engineering
train_x <- mutate(train_x,flag = 0)
test_x <- mutate(test_x,flag = 1)
train_total <- rbind(train_x, test_x)


# Data Cleaning by removing/imputing NaN values, factoring variables, 
#creating new useful features from existing variables and grouping variables 
train_total <- train_total %>% 
  mutate(
    security_deposit = parse_number(security_deposit),
    security_deposit = ifelse(is.na(security_deposit), median(security_deposit, na.rm = TRUE), security_deposit),
    host_since = as.Date(host_since),
    years_as_host = as.numeric(difftime(Sys.Date(), host_since, units = "days")) / 365,
    years_as_host = ifelse(is.na(years_as_host),median(years_as_host, na.rm = TRUE),years_as_host),
    availability_avg = (availability_30 + availability_60 + availability_90 + availability_365) / 4,
    availability_avg <- ifelse(is.na(availability_avg),median(availability_avg, na.rm = TRUE),min(availability_avg, na.rm = TRUE) /
                                 (max(availability_avg, na.rm = TRUE) - min(availability_avg, na.rm = TRUE))),
    
    first_review = as.Date(first_review),  # Convert first_review to date format
    property_first_used = as.numeric(difftime(Sys.Date(), first_review, units = "days")) / 365,
    property_first_used = ifelse(is.na(property_first_used),median(property_first_used, na.rm = TRUE),property_first_used),
    price = parse_number(price),
    availability_30_ratio = availability_30/30,
    average_nights_available =(maximum_nights + minimum_nights) / 2 ,
    booking_flexibility = ifelse(minimum_nights <= 2 & maximum_nights >= 7, "Flexible", "Not Flexible"),
    booking_flexibility = ifelse(is.na(booking_flexibility),"Not Flexible",booking_flexibility),
    property_category = if_else(property_type %in% c("Apartment", "Serviced apartment", "Loft"), "apartment",
                                if_else(property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel"), "hotel",
                                        if_else(property_type %in% c("Townhouse", "Condominium"), "condo",
                                                if_else(property_type %in% c("Bungalow", "House"), "house",
                                                        if_else(property_type %in% c("Resort", "Vacation home"), "vacation",
                                                                if_else(property_type %in% c("Castle", "Island", "Lighthouse"), "unique",
                                                                        if_else(property_type %in% c("Cottage", "Farm stay"), "country",
                                                                                if_else(property_type %in% c("Camper/RV", "Boat", "Train", "Plane"), "transport",
                                                                                        if_else(property_type %in% c("Guesthouse", "In-law", "Guest suite"), "guest",
                                                                                                if_else(property_type %in% c("Tent", "Dorm", "Hut", "Tipi"), "camping",
                                                                                                        "other")))))))))),
    property_category = factor(property_category),
    price_per_person = price / accommodates,
    ppp_ind = ifelse(price_per_person > tapply(price_per_person, property_category, median)[property_category], 1, 0),
    ppp_ind = factor(ppp_ind)
  )

train_total <- train_total %>%
  mutate(
    price = ifelse(price == 0, 1, price),
    price_category = factor(ifelse(price < 72, "Low",
                                   ifelse(price >= 72 & price < 150, "Medium",
                                          ifelse(price >= 150 & price < 175, "High",
                                                 ifelse(price >= 175, "Very High", "Unknown"))))),
    cancellation_policy = as.factor(ifelse(cancellation_policy == "super_strict_30", "strict",
                                           ifelse(cancellation_policy == "super_strict_60", "strict",
                                                  ifelse(cancellation_policy=="no_refunds", "strict",cancellation_policy)))),
    cleaning_fee = parse_number(cleaning_fee),
    cleaning_fee = ifelse(is.na(cleaning_fee), 0, cleaning_fee),
    has_cleaning_fee = as.factor(ifelse(cleaning_fee == 0, "NO", "YES")),
    square_feet <- ifelse(is.na(square_feet),median(square_feet,na.rm=TRUE),square_feet),
    beds = ifelse(is.na(beds), mean(beds, na.rm = TRUE), beds),
    extra_people = parse_number(extra_people),
    charges_for_extra = factor(if_else(extra_people > 0, "YES", "NO")),
    guests_included = ifelse(is.na(guests_included), 0, guests_included),
    host_identity_verified = as.factor(ifelse(is.na(host_identity_verified), "FALSE", host_identity_verified)),
    host_is_superhost = as.factor(ifelse(is.na(host_is_superhost), "FALSE", host_is_superhost)),
    has_min_nights = factor(if_else(minimum_nights > 1, "YES", "NO")),
   )

market_counts <- table(train_total$market)
small_markets <- names(market_counts[market_counts < 300])

train_total <- train_total %>%
  mutate(
    host_response_rate = parse_number(host_response_rate),
    host_response = factor(if_else(is.na(host_response_rate), "MISSING", if_else(host_response_rate == 100, "ALL", "SOME"))),
    instant_bookable = factor(instant_bookable),
    is_location_exact = factor(is_location_exact),
    require_guest_phone_verification = factor(require_guest_phone_verification),	
    requires_license = factor(requires_license),	
    room_type = factor(room_type),
    host_acceptance_rate = parse_number(host_acceptance_rate),
    host_acceptance = factor(if_else(is.na(host_acceptance_rate), "MISSING", if_else(host_acceptance_rate == 100, "ALL", "SOME"))),
    bathrooms = ifelse(is.na(bathrooms), mean(bathrooms, na.rm = TRUE), bathrooms),
    across(c("bedrooms", "square_feet"), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)),
    market = as.factor(ifelse(market %in% small_markets | is.na(market), "OTHER", market)),
)


# Analyzing features and insights

# 1: Square Feet Analysis: Histogram
# Create the scatter plot
ggplot(train_total, aes(x = square_feet, y = price)) +
  geom_point() +
  labs(x = "Square Feet", y = "Price") +
  ggtitle("Scatter Plot of Square Feet vs. Price")


# 2. Analysis of price category variable
level_order <- c("Low", "Medium", "High", "Very High")
train_total$price_category <- factor(train_total$price_category, levels = level_order)

# Count the number of listings in each price category
price_count <- table(train_total$price_category)

# Convert the result to a data frame
price_count_df <- data.frame(Price_Category = names(price_count),
                             Count = as.numeric(price_count))

price_count_df <- price_count_df[match(level_order, price_count_df$Price_Category), ]

# Plot the bar chart
ggplot(price_count_df, aes(x = Price_Category, y = Count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  xlab("Price Category") +
  ylab("Number of Listings") +
  ggtitle("Number of Listings by Price Category")


# 3. Line Chart for yearly number of hosts increasing
# Group the data by year and calculate the count of property_first_used
property_first_used_count <- train_total %>%
  group_by(year = lubridate::year(first_review)) %>%
  summarise(count = n())

# Create a line chart
ggplot(property_first_used_count, aes(x = year, y = count)) +
  geom_line() +
  labs(x = "Year", y = "Count of First Review Written")
scale_x_continuous(breaks = seq(min(property_first_used_count$year), max(property_first_used_count$year), by = 1))


#4. Market Analysis
market_counts <- table(train_total$market)
market_counts_df <- data.frame(Market = names(market_counts), Count = as.numeric(market_counts))

# Sort the data frame by count in descending order
market_counts_df <- market_counts_df[order(market_counts_df$Count, decreasing = TRUE), ]

# Create the bar chart
bar_chart <- ggplot(market_counts_df, aes(x = Market, y = Count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "Market", y = "Count") +
  ggtitle("Number of Listings by Market") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Display the bar chart
print(bar_chart)


# 5. Pie chart for Host response rate
# Calculate the count of host responses
response_count <- table(train_total$host_response)

# Calculate the percentage of host responses relative to the total count
response_percentage <- prop.table(response_count) * 100

# Create a pie chart
pie(response_percentage, labels = paste(names(response_percentage), "(", round(response_percentage, 1), "%)"), main = "Host Response Percentage")


# 6. Average availability in days by property category
# Group the dataset by property category and calculate average availability
availability_avg_by_category <- train_total %>%
  group_by(property_category) %>%
  summarize(avg_availability = mean(availability_avg, na.rm = TRUE))

# Create a bar chart or grouped bar chart
ggplot(availability_avg_by_category, aes(x = property_category, y = avg_availability)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(x = "Property Category", y = "Average Availability") +
  ggtitle("Average Availability by Property Category") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# 7. Cleaning fee impact on the Price
# Group the data by has_cleaning_fee and calculate the average price
cleaning_fee_data <- train_total %>%
  group_by(has_cleaning_fee) %>%
  summarize(avg_price = mean(price, na.rm = TRUE))

# Create the bubble chart
ggplot(cleaning_fee_data, aes(x = has_cleaning_fee, y = avg_price, size = avg_price, label = paste0("$", round(avg_price, 2)))) +
  geom_point(color = "blue", alpha = 0.6) +
  labs(x = "Has Cleaning Fee", y = "Average Price", title = "Average Price Comparison: Cleaning Fee") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold")) +
  geom_text(size = 4, nudge_y = 5) +
  scale_size(range = c(4, 12), breaks = seq(20, ceiling(max(cleaning_fee_data$avg_price)), 20))


#8. Distribution of Hosts based on years of Hosting
ggplot(train_total, aes(x = years_as_host)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Hosts based on Years as Host",
       x = "Years as Host",
       y = "Frequency")


#9.Bar chart showing the count of listings by Cancellation policy and Average Price for that
cancellation_prices <- train_total %>%
  group_by(cancellation_policy) %>%
  summarise(avg_price = mean(price))

cancellation_counts <- train_total %>%
  count(cancellation_policy) %>%
  arrange(desc(n))

ggplot(cancellation_counts, aes(x = cancellation_policy, y = n, fill = cancellation_prices$avg_price)) +
  geom_bar(stat = "identity", width=0.7) +
  geom_text(aes(label = n), vjust = -0.5, color = "black", size = 2.5) +
  scale_fill_viridis(option = "A", direction = 1) +
  scale_y_continuous(breaks = seq(0, max(cancellation_counts$n), by = 7000)) +
  labs(title = "Average Price Analysis by Cancellation Policy",
       x = "Cancellation Policy",
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))


# 10. Calculate the proportion of Listings with exact location information
location_exact_counts <- train_total %>%
  count(is_location_exact)
ggplot(location_exact_counts, aes(x = "", y = n, fill = is_location_exact)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  labs(title = "Proportion of Listings with Exact Location",
       fill = "Is Location Exact") +
  
  scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "darkorange")) +
  theme_minimal() +
  theme(axis.title = element_blank(),
        axis.text = element_blank(),
        panel.grid = element_blank(),
        legend.title = element_blank())


#Splitting train and test data after cleaning
new_train_x <- filter(train_total, flag == 0)
new_test_x <- filter(train_total, flag == 1)


#Combining train data with train target variable
train <- cbind(new_train_x, train_y) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score),
         high_booking_rate = as.factor(high_booking_rate)) 

train_perfect <- train %>%
  select(-high_booking_rate)


# Create dummy for Train using feature features
airbnb_features <- train_perfect %>%
  select(price_category,booking_flexibility,availability_30_ratio,square_feet,average_nights_available,property_first_used,availability_avg,years_as_host,accommodates,bedrooms, ppp_ind, beds, cancellation_policy, has_cleaning_fee, charges_for_extra, host_identity_verified,market, host_is_superhost, host_response, instant_bookable , has_min_nights, price, guests_included, property_category ,security_deposit, is_location_exact, host_acceptance, bathrooms, require_guest_phone_verification, requires_license, room_type)

dummy_train <- dummyVars( ~ . , data=airbnb_features, fullRank = TRUE)
train_dummy_df <- data.frame(predict(dummy_train, newdata = airbnb_features)) 
train_dummy_df$perfect_rating_score = train$perfect_rating_score

#Splitting the training dataset to train and validation for evaluating the model performance
train_insts <- sample(nrow(train_dummy_df), 0.7 * nrow(train_dummy_df))
data_train <- train_dummy_df[train_insts, ]
data_valid <- train_dummy_df[-train_insts, ]

#Baseline model Accuracy
summary(data_train$perfect_rating_score)
baseline_preds = rep("NO", nrow(data_valid)) 
baseline_correct = ifelse(baseline_preds == data_valid$perfect_rating_score, 1, 0)
baseline_accuracy =  sum(baseline_correct)/length(baseline_correct)
baseline_accuracy

#baseline ends


#Random forest **********************
rf.mod <- ranger(perfect_rating_score~ . , data = data_train,
                 mtry=20, num.trees=800,
                 importance="impurity",
                 probability = TRUE)

#Generalization Performance of Ranger
rf_preds <- predict(rf.mod, data=data_valid)$predictions[,2]
rf_classifications <- ifelse(rf_preds>0.5048, "YES", "NO")
rf_acc <- mean(rf_classifications == data_valid$perfect_rating_score)
rf_acc

cm_rf <- table(rf_classifications, data_valid$perfect_rating_score)
print(cm_rf)

TPR_rf <- cm_rf[2,2] / sum(cm_rf[,2])
FPR_rf <- cm_rf[2,1] / sum(cm_rf[,1])

print(paste0("Ranger TPR: ", TPR_rf))
print(paste0("Ranger FPR:", FPR_rf))

#Training Performance of Ranger
rf_preds_train <- predict(rf.mod, data=data_train)$predictions[,2]
rf_classifications_train <- ifelse(rf_preds_train>0.5048, "YES", "NO")
rf_acc_train <- mean(rf_classifications_train == data_train$perfect_rating_score)
rf_acc_train

cm_rf_train <- table(rf_classifications_train, data_train$perfect_rating_score)
print(cm_rf_train)

TPR_rf_train <- cm_rf_train[2,2] / sum(cm_rf_train[,2])
FPR_rf_train <- cm_rf_train[2,1] / sum(cm_rf_train[,1])

print(paste0("Ranger TPR (Training Performance): ", TPR_rf_train))
print(paste0("Ranger FPR (Training Performance): ", FPR_rf_train))


# ****************************************


# Logistic Regression **********************
logistic_perfect <- glm(perfect_rating_score ~ .,data = data_train, family = "binomial")

#Generalization Performance of Logistic
pred_logistic <- predict(logistic_perfect, newdata = data_valid, type = "response")
pred_class_logistic <- ifelse(pred_logistic > 0.480, "YES", "NO")
accuracy_logistic <- mean(pred_class_logistic == data_valid$perfect_rating_score)
print(accuracy_logistic)

cm_logistic <- table(pred_class_logistic, data_valid$perfect_rating_score)
print(cm_logistic)

TPR_logistic <- cm_logistic[2,2] / sum(cm_logistic[,2])
FPR_logistic <- cm_logistic[2,1] / sum(cm_logistic[,1])

print(paste0("Logistic TPR: ", TPR_logistic))
print(paste0("Logistic FPR: ", FPR_logistic))

#Training Performance of Logistic
pred_logistic_train <- predict(logistic_perfect, newdata = data_train, type = "response")
pred_class_logistic_train <- ifelse(pred_logistic_train > 0.480, "YES", "NO")
accuracy_logistic_train <- mean(pred_class_logistic_train == data_train$perfect_rating_score)
print(accuracy_logistic_train)

cm_logistic_train <- table(pred_class_logistic_train, data_train$perfect_rating_score)
print(cm_logistic_train)

TPR_logistic_train <- cm_logistic_train[2,2] / sum(cm_logistic_train[,2])
FPR_logistic_train <- cm_logistic_train[2,1] / sum(cm_logistic_train[,1])

print(paste0("Logistic TPR (Training Performance): ", TPR_logistic_train))
print(paste0("Logistic FPR (Training Performance): ", FPR_logistic_train))


# ****************************************


# Decision Tree **********************
perfect_tree <- rpart(perfect_rating_score ~ ., data = data_train, control= c(cp = 0.0001, maxdepth = 8) , method="class")

#Generalization Performance of Decision Tree
pred_tree <- predict(perfect_tree, newdata = data_valid, type = "prob")[,2]
pred_class_tree <- ifelse(pred_tree > 0.478, "YES", "NO")

accuracy_tree <- mean(pred_class_tree == data_valid$perfect_rating_score)
print(accuracy_tree)

cm_tree <- table(pred_class_tree, data_valid$perfect_rating_score)
print(cm_tree)

TPR_tree <- cm_tree[2,2] / sum(cm_tree[,2])
FPR_tree <- cm_tree[2,1] / sum(cm_tree[,1])

print(paste0("Decision Tree TPR: ", TPR_tree))
print(paste0("Decision Tree FPR: ", FPR_tree))

#Training Performance of Decision Tree
pred_tree_train <- predict(perfect_tree, newdata = data_train, type = "prob")[,2]
pred_class_tree_train <- ifelse(pred_tree_train > 0.478, "YES", "NO")

accuracy_tree_train <- mean(pred_class_tree_train == data_train$perfect_rating_score)
print(accuracy_tree_train)

cm_tree_train <- table(pred_class_tree_train, data_train$perfect_rating_score)
print(cm_tree_train)

TPR_tree_train <- cm_tree_train[2,2] / sum(cm_tree_train[,2])
FPR_tree_train <- cm_tree_train[2,1] / sum(cm_tree_train[,1])

print(paste0("Decision Tree TPR (Training Performance): ", TPR_tree_train))
print(paste0("Decision Tree FPR (Training Performance): ", FPR_tree_train))

# ****************************************

# Ridge Regression********************** 
ridge_x <- data.matrix(data_train[, -ncol(data_train)])
ridge_y <- data_train$perfect_rating_score

valid_y <- data_valid$perfect_rating_score

accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}

grid <- 10^seq(-7, 7, length.out = 100)
accs <- rep(0, length(grid))

for(i in c(1:length(grid))){
  lam = grid[i] #current value of lambda
  
  #train a ridge model with lambda = lam
  ridge_cv <- glmnet(ridge_x, ridge_y, family="binomial",
                     alpha = 0, lambda = lam)
  
  #make predictions as usual
  pred_ridge <- predict(ridge_cv, newx = as.matrix(data_valid[, -ncol(data_valid)]), type = "response")
  
  #classify and compute accuracy
  classification <- ifelse(pred_ridge > 0.481, "YES", "NO")
  inner_acc <- accuracy(classification, valid_y)
  accs[i] <- inner_acc
}

#plot fitting curve
plot(log10(grid), accs)

best_validation_index <- which.max(accs)
best_lambda_ridge <- grid[best_validation_index]

cat("Optimal Lambda:")
best_lambda_ridge

ridge_fit <- glmnet(ridge_x, ridge_y, family="binomial",
                    alpha = 0, lambda = best_lambda_ridge)

#Generalization Performance of Ridge
preds_ridge <- predict(ridge_fit, newx =as.matrix(data_valid[, -ncol(data_valid)]), type = "response")
class_ridge <- ifelse(preds_ridge > 0.481, "YES", "NO")

accuracy_ridge <- mean(class_ridge == data_valid$perfect_rating_score)
print(accuracy_ridge)

cm_ridge <- table(class_ridge, data_valid$perfect_rating_score)
print(cm_ridge)

TPR_ridge <- cm_ridge[2,2] / sum(cm_ridge[,2])
FPR_ridge <- cm_ridge[2,1] / sum(cm_ridge[,1])

print(paste0("Ridge TPR: ", TPR_ridge))
print(paste0("Ridge FPR: ", FPR_ridge))

#Training Performance of Ridge
preds_ridge_train <- predict(ridge_fit, newx =as.matrix(data_train[, -ncol(data_train)]), type = "response")
class_ridge_train <- ifelse(preds_ridge_train > 0.481, "YES", "NO")

accuracy_ridge_train <- mean(class_ridge_train == data_train$perfect_rating_score)
print(accuracy_ridge)

cm_ridge_train <- table(class_ridge_train, data_train$perfect_rating_score)
print(cm_ridge_train)

TPR_ridge_train <- cm_ridge_train[2,2] / sum(cm_ridge_train[,2])
FPR_ridge_train <- cm_ridge_train[2,1] / sum(cm_ridge_train[,1])

print(paste0("Ridge TPR (Training Performance): ", TPR_ridge_train))
print(paste0("Ridge FPR (Training Performance): ", FPR_ridge_train))

# ****************************************

# Lasso Regression **********************
lasso_x <- data.matrix(data_train[, -ncol(data_train)])
lasso_y <- data_train$perfect_rating_score

grid <- 10^seq(-7, 7, length.out = 100)
accs <- rep(0, length(grid))

for(i in c(1:length(grid))){
  lam = grid[i] #current value of lambda
  
  lasso_cv <- glmnet(lasso_x, lasso_y, family="binomial",
                     alpha = 1, lambda = lam)
  
  #make predictions as usual
  pred_lasso <- predict(lasso_cv, newx = as.matrix(data_valid[, -ncol(data_valid)]), type = "response")
  
  #classify and compute accuracy
  classification <- ifelse(pred_lasso > 0.481, "YES", "NO")
  inner_acc <- accuracy(classification, valid_y)
  accs[i] <- inner_acc
}

plot(log10(grid), accs)

best_validation_index <- which.max(accs)
best_lambda_lasso <- grid[best_validation_index]

cat("Optimal Lambda:")
best_lambda_lasso

lasso_fit <- glmnet(lasso_x, lasso_y, family="binomial",
                    alpha = 1, lambda = best_lambda_lasso)

#Generalization Performance of Lasso
preds_lasso <- predict(lasso_fit, newx =as.matrix(data_valid[, -ncol(data_valid)]), type = "response")
class_lasso <- ifelse(preds_lasso > 0.481, "YES", "NO")

accuracy_lasso <- mean(class_lasso == data_valid$perfect_rating_score)
print(accuracy_lasso)

cm_lasso <- table(class_lasso, data_valid$perfect_rating_score)
print(cm_lasso)

TPR_lasso <- cm_lasso[2,2] / sum(cm_lasso[,2])
FPR_lasso <- cm_lasso[2,1] / sum(cm_lasso[,1])

print(paste0("Lasso TPR: ", TPR_lasso))
print(paste0("Lasso FPR: ", FPR_lasso))

#Training Performance of Lasso
preds_lasso_train <- predict(lasso_fit, newx =as.matrix(data_train[, -ncol(data_train)]), type = "response")
class_lasso_train <- ifelse(preds_lasso_train > 0.481, "YES", "NO")

accuracy_lasso_train <- mean(class_lasso_train == data_train$perfect_rating_score)
print(accuracy_lasso_train)

cm_lasso_train <- table(class_lasso_train, data_train$perfect_rating_score)
print(cm_lasso_train)

TPR_lasso_train <- cm_lasso_train[2,2] / sum(cm_lasso_train[,2])
FPR_lasso_train <- cm_lasso_train[2,1] / sum(cm_lasso_train[,1])

print(paste0("Lasso TPR (Training Performance): ", TPR_lasso_train))
print(paste0("Lasso FPR (Training Performance): ", FPR_lasso_train))

# ****************************************

# Logistic ROC Curve
prediction_logistic <- prediction(pred_logistic, valid_y)
roc_logistic <- performance(prediction_logistic, "tpr", "fpr")
plot(roc_logistic, col = "green", lwd = 2)
legend("bottomright", legend = "Logistic", col = "green", lty = 1, lwd = 2, bg = "white")

# Lasso ROC Curve
prediction_lasso <- prediction(preds_lasso, valid_y)
roc_lasso <- performance(prediction_lasso, "tpr", "fpr")
plot(roc_lasso, add=TRUE, col = "cyan", lwd = 2)
legend("bottomright", legend = "Lasso", col = "cyan", lty = 1, lwd = 2, bg = "white")

#plotting ROC Curves for Random Forest, Ridge Regression and Decision Tree
prediction_rf <- prediction(rf_preds, valid_y)
prediction_ridge <- prediction(preds_ridge, valid_y)
prediction_trees <- prediction(pred_tree, valid_y)

roc_rf <- performance(prediction_rf, "tpr", "fpr")
roc_trees <- performance(prediction_trees, "tpr", "fpr")
roc_ridge <- performance(prediction_ridge, "tpr", "fpr")

plot(roc_rf, col = "blue", lwd = 2)
plot(roc_trees, add=TRUE, col = "orange", lwd = 2)
plot(roc_ridge, add=TRUE, col = "red", lwd = 2)

legend("bottomright", 
       legend=c( "Ranger" , "Decision Trees", "Ridge"),
       col=c("blue", "orange", "red"),
       lty=1, lwd=2, bg="white")

# Learning curve - Ranger Random Forest
train_sizes <- seq(0.1, 1, by = 0.1)  # Varying training set sizes

rf_accuracy <- vector()  # Vector to store accuracy values

# Adjust figure margins
par(mar = c(5, 5, 4, 2) + 0.1)

# Iterate over different training set sizes
for (size in train_sizes) {
  n_samples <- floor(nrow(data_train) * size)
  train_subset <- data_train[sample(nrow(data_train), n_samples), ]
  
  rf_subset <- ranger(perfect_rating_score ~ ., data = train_subset,
                      mtry = 20, num.trees = 800,
                      importance = "impurity",
                      probability = TRUE)
  
  rf_subset_preds <- predict(rf_subset, data = data_valid)$predictions[, "YES"]
  rf_subset_classifications <- as.numeric(rf_subset_preds > 0.5)
  
  rf_subset_acc <- mean(rf_subset_classifications == as.numeric(data_valid$perfect_rating_score == "YES"))
  rf_accuracy <- c(rf_accuracy, rf_subset_acc)
}

# Adjust figure margins
par(mar = c(5, 5, 4, 2) + 0.1)

# Create a learning curve plot
plot(train_sizes, rf_accuracy, type = "b", pch = 16,
     xlab = "Training Set Size", ylab = "Accuracy",
     main = "Learning Curve - Random Forest")

# Create dummy for Test 
airbnb_features2 <- new_test_x %>%
  select(price_category,booking_flexibility,availability_30_ratio,square_feet,average_nights_available,property_first_used,availability_avg,years_as_host,accommodates,bedrooms, ppp_ind, beds, cancellation_policy, has_cleaning_fee, charges_for_extra, host_identity_verified,market, host_is_superhost, host_response, instant_bookable , has_min_nights, price, guests_included, property_category ,security_deposit, is_location_exact, host_acceptance, bathrooms, require_guest_phone_verification, requires_license, room_type)

dummy_test <- dummyVars( ~ . , data=airbnb_features2, fullRank = TRUE)
test_dummy_df <- data.frame(predict(dummy_test, newdata = airbnb_features2))
test_dummy_df$perfect_rating_score <- NA

#Use of Ranger Model for Predictions 
rf_perfect <- ranger(perfect_rating_score~ . , data = train_dummy_df, mtry=20, num.trees=800,
                     importance="impurity",
                     probability = TRUE)

probs_perfect <- predict(rf_perfect, data=test_dummy_df)$predictions[,2]

summary(rf_perfect)

#Classification of Data
classifications_perfect <- ifelse(probs_perfect>0.5048, "YES", "NO")

write.table(classifications_perfect, "perfect_rating_score_group4.csv", row.names = FALSE, col.names= 'perfect_rating_score')
