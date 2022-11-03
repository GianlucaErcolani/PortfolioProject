# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:37:37 2022

@author: gianl
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler


## Data Import and Exploration

df = pd.read_csv(r"C:\Users\gianl\Downloads\diabetes (2).csv")

df.head()

df.info()

df.describe()

# Distributions for all variables 

df.hist()

# Pairplot
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=14)   
sns.set_style('darkgrid')
sns.pairplot(df,  plot_kws={'color':'black'})


# Separate target and features

x = df.iloc[:, 0:8]
y = df.iloc[:, 8]

# Scaling

scale = StandardScaler()

scaled_x= scale.fit_transform(x)

scaled_x = pd.DataFrame(scaled_x)

scaled_x.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

scaled_x

# Correlation matrix

scaled_x.corr()
sns.heatmap(scaled_x.corr(), annot=True, cmap="Blues")

 
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

scaled_x_train, scaled_x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.25, random_state=16)

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(scaled_x_train, y_train)


# Measuring accuracy through cross validation. Cross validation is a resampling method. In this case I use 5-fold cross-validation.  K-fold cross-validation means splitting the training set into 5 folds, then training the model 5 times, holding out a different fold each time for evaluation.

from sklearn.model_selection import cross_val_score
cross_val_score(logreg, scaled_x_train, y_train, cv=5, scoring="accuracy")

from sklearn.model_selection import cross_val_predict

y_pred_cv = cross_val_predict(logreg, scaled_x_train, y_train, cv=5)


# Confusion matrix. It is a useful tool for summarizing how a method performed on the testing data and for discriminating among different methods, through the calculation of several metrics.
# Metrics like sensitivity, specificity, ROC and AUC.

# Importing the metrics package from sklearn library
from sklearn import metrics
# Creating the confusion matrix
cm = metrics.confusion_matrix(y_train, y_pred_cv)
# Assigning columns names
cm_df = pd.DataFrame(cm, 
            columns = ['Predicted Negative', 'Predicted Positive'],
            index = ['Actual Negative', 'Actual Positive'])
# Showing the confusion matrix
cm_df


# Importing the metrics package from sklearn library
from sklearn import metrics
# Creating the confusion matrix
cm = metrics.confusion_matrix(y_train, y_pred_cv)
# Assigning columns names
cm_df = pd.DataFrame(cm, 
            columns = ['Predicted Negative', 'Predicted Positive'],
            index = ['Actual Negative', 'Actual Positive'])
# Showing the confusion matrix
cm_df


# Creating a function to report confusion metrics
def confusion_metrics (conf_matrix):
# save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
    conf_misclassification = 1 - conf_accuracy
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    print('-'*50)
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}') 
    print(f'Precision: {round(conf_precision,2)}')
    print(f'f_1 Score: {round(conf_f1,2)}')


confusion_metrics(cm)

# Accuracy 0.77, sensitivity 0.56 and specificity 0.89



# Calculating class probabilities
pred_proba = [i[1] for i in logreg.predict_proba(scaled_x_train)]
pred_df = pd.DataFrame({'true_values': y_train,
                        'pred_probs':pred_proba})



# ROC (Receiver Operator Characteristic) curve is useful for identifying the best threshold for making a decision.
# I've drawn ROC graph using True Positive Rates (Sensitivity) and False positive Rates (1 - Specificity). It can be drawn in other ways, like replacing the False Positive Rate with Precision(the proportion of positive results that were correctly classified).
# AUC (Area under the curve) is useful for discriminate among machine learning techniques.

# Create figure.
plt.figure(figsize = (10,7))
# Create threshold values. 
thresholds = np.linspace(0, 1, 200)
# Define function to calculate sensitivity. (True positive rate.)
def TPR(df, true_col, pred_prob_col, threshold):
    true_positive = df[(df[true_col] == 1) & (df[pred_prob_col] >= threshold)].shape[0]
    false_negative = df[(df[true_col] == 1) & (df[pred_prob_col] < threshold)].shape[0]
    return true_positive / (true_positive + false_negative)
# Define function to calculate 1 - specificity. (False positive rate.)
def FPR(df, true_col, pred_prob_col, threshold):
    true_negative = df[(df[true_col] == 0) & (df[pred_prob_col] <= threshold)].shape[0]
    false_positive = df[(df[true_col] == 0) & (df[pred_prob_col] > threshold)].shape[0]
    return 1 - (true_negative / (true_negative + false_positive))
# Calculate sensitivity & 1-specificity for each threshold between 0 and 1.
tpr_values = [TPR(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
fpr_values = [FPR(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
# Plot ROC curve.
plt.plot(fpr_values, # False Positive Rate on X-axis
         tpr_values, # True Positive Rate on Y-axis
         label='ROC Curve')
# Plot baseline. (Perfect overlap between the two populations.)
plt.plot(np.linspace(0, 1, 200),
         np.linspace(0, 1, 200),
         label='baseline',
         linestyle='--')
# Label axes.
plt.title(f"ROC Curve with AUC = {round(metrics.roc_auc_score(pred_df['true_values'], pred_df['pred_probs']),3)}", fontsize=22)
plt.ylabel('Sensitivity', fontsize=18)
plt.xlabel('1 - Specificity', fontsize=18)
# Create legend.
plt.legend(fontsize=16);




# Predicted value without cross validation
y_pred = logreg.predict(scaled_x_test)


# Creating the confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
# Assigning columns names
cm_df = pd.DataFrame(cm, 
            columns = ['Predicted Negative', 'Predicted Positive'],
            index = ['Actual Negative', 'Actual Positive'])
# Showing the confusion matrix
cm_df

# Metrics of the confusion matrix
confusion_metrics(cm)

# Accuracy 0.82, Sensitivity 0.61 and specificity 0.93

# Calculating class probabilities
pred_proba = [i[1] for i in logreg.predict_proba(scaled_x_test)]
pred_df = pd.DataFrame({'true_values': y_test,
                        'pred_probs':pred_proba})

# Create figure.
plt.figure(figsize = (10,7))
# Create threshold values. 
thresholds = np.linspace(0, 1, 200)

# Calculate sensitivity & 1-specificity for each threshold between 0 and 1.
tpr_values = [TPR(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
fpr_values = [FPR(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
# Plot ROC curve.
plt.plot(fpr_values, # False Positive Rate on X-axis
         tpr_values, # True Positive Rate on Y-axis
         label='ROC Curve')
# Plot baseline. (Perfect overlap between the two populations.)
plt.plot(np.linspace(0, 1, 200),
         np.linspace(0, 1, 200),
         label='baseline',
         linestyle='--')
# Label axes.
plt.title(f"ROC Curve with AUC = {round(metrics.roc_auc_score(pred_df['true_values'], pred_df['pred_probs']),3)}", fontsize=22)
plt.ylabel('Sensitivity', fontsize=18)
plt.xlabel('1 - Specificity', fontsize=18)
# Create legend.
plt.legend(fontsize=16);






