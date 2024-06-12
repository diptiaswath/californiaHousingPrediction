**Problem Statement:**

Build a Regression Model to predict the price of a house in California. Target variable to predict = median_house_value. Also, interpret the resulting "best" model.

**Summary of Results:**

- Resulting "best" model is a Linear Regression Model  (train/test = 70/30) with degree = 3.  
- This gives the smallest MSE (Mean Squared Error) on the test data set, where MSE= 3.96929E+9.  
- For this Degree 3 model, we have latitude and longitude as variables that contributes the most to this model’s performance, followed by variables- total_bedrooms, population and median_income contributing as well. This model has a R^2 score of 0.6885 with the test data set, and a R^2 score of 0.7228 with the training data set. 

**** Detailed Analysis: ****
Dataset Description: Housing Dataset provided has below columns with a total of 20640 entries. 

Data columns (total 10 columns):

#   Column              Non-Null Count  Dtype 
---  ------              --------------  ----- 
0   longitude           20640 non-null  float64
1   latitude            20640 non-null  float64
2   housing_median_age  20640 non-null  float64
3   total_rooms         20640 non-null  float64
4   total_bedrooms      20433 non-null  float64
5   population          20640 non-null  float64
6   households          20640 non-null  float64
7   median_income       20640 non-null  float64
8   median_house_value  20640 non-null  float64
9   ocean_proximity     20640 non-null  object
dtypes: float64(9), object(1)

memory usage: 1.6+ MB

**Data Preprocessing:**

Since target variable to predict is 'median_house_value', we have X and y as below: X = cali.drop(columns='median_house_value') y = cali.median_house_value
ocean_proximity is a category column with unique values of ['NEAR BAY' '<1H OCEAN' 'INLAND' 'NEAR OCEAN' 'ISLAND’].
total_bedrooms is identified as a column with missing values.
Treated multi-collinearity in this data-set with VIF (Variance Inflation Factor). Dropped households and total_rooms as features with high VIF (>5)
**Before treating data-set for multi-collinearity:**

Module8.jpeg

**After treating data-set for multi-collinearity:**

Module8-1.jpeg

**Training/Test Data Split:**

Split the dataset (with 20640 entries) with a 70/30 split and random_state=22. Note, that median_house_value is dropped from X
Shape of train and test data-set with this split: X1_train.shape, X1_test.shape, y1_train.shape, y1_test.shape - ((14448, 7), (6192, 7), (14448,), (6192,))
Pipeline for data-processing tasks:

Pipeline uses the following column transformers - an ordinal encoder for ocean_proximity, a degree 1-5 polynomial transformer for numeric features, a simple imputer applied on column total_bedrooms with missing values, replaces with median value of this column, and a standard scaler to ensure each variable has a std dev of 1 with a mean of 0. Remainder of variables are passthrough
Module8-2.jpeg

**Built a Linear Regression Model for degrees 1 through 5, and identified optimal model**

Best degree polynomial model for degree = 3
This degree 3 model on the test data-set results in a MSE = 3969290239.71
Plot: Polynomial degree v.s. MSE for test and train data-set

newplot (1)-2.png

 

**Extracted Predicated and Actual Median House Values on the test data set**

Saved the predicted median home values for the test and train data-set for degree 3 model (optimal/best model) in two separate csv files (attached) - data/train_with_predictions_for_best_model.csv and data/train_with_predictions_for_best_model.csv test_with_predictions_for_best_model.csv Download test_with_predictions_for_best_model.csvtrain_with_predictions_for_best_model.csv Download train_with_predictions_for_best_model.csv 
Plot: Actual v.s. Predicated value for median_house_value for degree 3 model on Test Data

newplot-3.png

**Calculated a baseline linear regression model score**

**Observed :** Training R^2 score with Baseline model = 0.6292599613542642, Test R^2 score with Baseline model = 0.6407365503098301
Evaluated Permutation Feature Importance of a Degree3 (Best Complexity/Optimal Model) and a Degree5 Model

For the best complexity/optimal model with degree 3, observed the model scores:
Training R^2 score with degree 3 model: 0.7228312698858961

Test R^2 score with degree 3 model: 0.6885230431826338

Feature longitude: Importance = 2.4226551609317304

Feature latitude: Importance = 3.2346079024550534

Feature housing_median_age: Importance = 0.08299161544168861

Feature total_bedrooms: Importance = 1.419046012724737

Feature population: Importance = 1.129178859699262

Feature median_income: Importance = 0.8273510118685123

Feature ocean_proximity: Importance = 0.00398225589478225

Conclusion: Interpreting the model with permutation feature importance - 

For this Degree 3 model, we have latitude and longitude as variables that contributes the most to this model’s performance, followed by the total_bedrooms, population and median_income contributing as well.  Below is a Plot that shows how each variable/feature contributes to this model’s performance.
newplot (2)-2.png

**Compared and Contrasted with a degree 5 model**, the test data gives a negative model score, while the training data has almost an equivalent model score than its degree 3 model. It’s also interesting to note, we have population contributing the most to this model’s performance. The negative model score indicates this model performs poorly with test data. This implies the true relationship between the target variable median_house_value and the other independent features are not captured well.
Training R^2 score with degree 5 model: 0.7379391078800335

Test R^2 score with degree 5 model: -0.06560209255343108

Feature longitude: Importance = 3.304637719117866

Feature latitude: Importance = 5.420857258615869

Feature housing_median_age: Importance = 2.9464260001819715

Feature total_bedrooms: Importance = 295.59155975717425

Feature population: Importance = 658.1235050357736

Feature median_income: Importance = 1.632526924755118

Feature ocean_proximity: Importance = 3.0471700052103755e-09

newplot (3)-1.png

