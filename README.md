# ML_Health1_Sepsis_AKI
2019 PhysioNet Challenge for Sepsis and AKI Prediction

## Setup
```
pip install -r requirements.txt
```
You need to download the Sepsis and AKI prediction data and put it into
a folder called 'data'. Within data, you have two subdirectories for aki and
sepsis. And then within each of these directories, you have two more
subdirectories of train and test. For example:
data > sepsis > train > your csv files

Likewise, you need a folder called 'results'. This is where the scripts will
output the files.

## Programs to run in order for predictions of AKI / Sepsis
### 1. Generate a csv of the distribution of the 35 continuous features
Show the distribution of the 35 continuous features for the sepsis and
AKI training dataset, as well as a count of missing values. 
```
python create_distribution.py --dataset [sepsis|aki]
```
Returns file: {dataset}_features.csv 

### 2. Create the Labels
Creates a table (.csv file) describing whether a patient in the training dataset
develops AKI or Sepsis at any point during their hospital stay
```
python create_labels.py --dataset [sepsis|aki]
```
Returns file: {dataset}.csv 

## Create the train and test prediciton matrix
Formats the data for the prediction models. It will create a matrix with size
number of patients x 35 features. In order to predict either sepsis or aki
on the test dataset, you must create a prediction matrix of both the train and
test data splits.

Note: onset_window is only required for the aki dataset
```
python create_prediction_matrix.py --dataset [sepsis|aki] --data_split[train|test] --onset_window[24|48|72]
```
Returns file: {data_split}_{dataset}_pred_matrix_{onset_window}.csv 

## Hypertune the training models for prediction
Find the best hyperparameters for both the Logistic Regression and Random Forest Models
```
python make_predictions.py --dataset [sepsis|aki] --onset_window[24|48|72] --find_hyperparameters True
```
Returns: Prints results to console

## Make Predictions
Returns the prediction performance, most important features, and test set predictions
```
python make_predictions.py --dataset [sepsis|aki] --onset_window[24|48|72] --find_hyperparameters False \
--smote [True|False] --C [0.1, 0.25, 0.5, 0.75, 1] --penalty [l1|l2] --n_estimators [10, 20, 50, 80, 100] \
--max_depth [5, 10, 15, 18, 20]
```
Returns 3 files: (1) {dataset}_train_results_{smote}.csv (2) {dataset}_features_imp_{smote}.csv, (3) {dataset}_pred_{smote}.csv

# Example of AKI - 24 Hour Window Prediction of Test Data Set
1. python create_labels.py --dataset aki
2. python create_distribution.py --dataset aki --onset_window 24 --data_split train
3. python create_distribution.py --dataset aki --onset_window 24 --data_split test
4. python make_predictions.py --dataset aki --onset_window 24 --find_hyperparameters True
Note: Take the hyperparameters found and input in the next call
with SMOTE
5. python make_predictions.py --dataset aki --onset_window 24 --find_hyperparameters False \
--smote True --C 0.1 --penalty l1 --n_estimators 80 \
--max_depth 18
without SMOTE:
6. python make_predictions.py --dataset aki --onset_window 24 --find_hyperparameters False \
--smote False --C 0.25 --penalty l1 --n_estimators 100 \
--max_depth 15

These 6 operations would create the following files in the results folder:
aki.csv
train_aki_pred_matrix_24.csv
test_aki_pred_matrix_24.csv
test_aki_pred_matrix_24.csv
aki_ow1_train_results.csv
aki_ow1_train_results_smote.csv
aki_ow1_features_imp.csv
aki_ow1_features_imp_smote.csv
aki_ow1_pred.csv
aki_ow1_pred_smote.csv

# Best Hyperparameters for the models
SEPSIS HYPERPARAMETERS:
SMOTE:  False
C= 0.1 , penalty= l1, n_estimators= 10 , max_depth= 20
SMOTE:  True
C= 0.1 , penalty= l2, n_estimators= 100 , max_depth= 20

AKI-24 HYPERPARAMETERS:
SMOTE:  False
C= 0.1, penalty= l2, n_estimators= 10, max_depth= 18
SMOTE:  True
C= 0.75, penalty= l2, n_estimators= 100, max_depth= 20

AKI-48 HYPERPARAMETERS:
SMOTE:  False
No good hyperparameters (all bad).
SMOTE:  True
C= 1, penalty= l2, n_estimators= 100, max_depth= 20

AKI-72 HYPERPARAMETERS:
SMOTE:  False
No good hyperparameters (all bad).
SMOTE:  True
C= 1, penalty= l1, n_estimators= 100, max_depth= 20