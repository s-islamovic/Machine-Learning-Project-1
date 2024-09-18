#!/usr/bin/env python
# coding: utf-8

# # A classification approach to predict the likelihood of an employee leaving a company

# # Importing Libraries and Packages

# In[83]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# # Data exploration and Preprocessing

# In[33]:


# importing our data set
df = pd.read_csv("HR-Employee-Attrition.csv")
df.head()


# In[56]:


df.shape


# In[59]:


df.nunique()


# In[34]:


# Summary Descriptive statistics
df.describe()


# In[35]:


df.info()


# In[36]:


# data cleaning - check for missing values
df.isnull().sum()


# In[37]:


# remove columns that are not relevant for our analysis
df = df.drop(['EmployeeCount','Over18','StandardHours', 'EmployeeNumber',], axis=1)
df


# In[38]:


# Creating a Box plot to spot for any outliers
plt.figure(figsize=(10,6))

plt.subplot(1, 2, 2)
sns.boxplot(data=df)
plt.title('Box Plot of All Features')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.xlabel('Features')
plt.ylabel('Values')

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


# In[39]:


# now we want to remove the outlier using IQR
Q1= df['MonthlyIncome'].quantile(0.25)
Q3= df['MonthlyIncome'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
clean_data=df[(df['MonthlyIncome']>=lower) & (df['MonthlyIncome']<=upper)]

clean_data.reset_index()


# # Exploratory Data Analysis (EDA)

# In[43]:


# Correlation matrix for all attributes

correlation= clean_data.select_dtypes(np.number).corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, fmt='.2f')


# In[45]:


# Now we create ancorrelation matrix with highly correlated attributes

correlation2 = clean_data[['Age','JobLevel','MonthlyIncome','NumCompaniesWorked','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation2,fmt='.2f',annot=True)
plt.xticks(rotation=40)
plt.show()


# In[46]:


# Now we are making a pairplot for the most correlated attributes

cols = ['TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','Attrition']
sns.pairplot(clean_data[cols], hue='Attrition')


# In[55]:


# Attrition correlation analysis

columns=['YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion']

fig,axes=plt.subplots(nrows=3,ncols=1,figsize=(14,10))
for i,col in enumerate(columns):
    sns.pointplot(x=clean_data[col], y=clean_data['TotalWorkingYears'],ax=axes[i], hue=clean_data['Attrition'])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('TotalWorkingYears')

plt.show()


# In[66]:


# To visualise attrition rates among job roles

plt.figure(figsize=(13,10))
sns.histplot(clean_data,x='JobRole',hue='Attrition')
plt.xticks(rotation=30)
plt.xlabel('Job Role', fontsize=13)
plt.ylabel('Count', fontsize=13)
plt.show()


# In[69]:


# Further checking for attrition relationships with attributes

plt.figure(figsize=(10,8))
sns.histplot(clean_data,x='DistanceFromHome',hue='Attrition')
plt.xticks(rotation=30)
plt.xlabel('Distance From Home (KM)', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.show()


# In[70]:


# No we want to check the relationship between Attrition and marital status 
clean_data.groupby('MaritalStatus').size().plot(kind='pie',autopct='%.2f')


# In[71]:


# We can tell by the graphs that there is not any significant correlation

married = clean_data[clean_data['MaritalStatus'] == 'Married']['Attrition']
divorced = clean_data[clean_data['MaritalStatus'] == 'Divorced']['Attrition']
single = clean_data[clean_data['MaritalStatus'] == 'Single']['Attrition']

fig,axes = plt.subplots(ncols=3, nrows=1,figsize=(14,10))
clean_data.groupby(married).size().plot(kind='pie',autopct='%.2f', ax=axes[0], title='Married Attrition')
clean_data.groupby(single).size().plot(kind='pie',autopct='%.2f', ax=axes[1], title='Single Attrition')
clean_data.groupby(divorced).size().plot(kind='pie',autopct='%.2f', ax=axes[2], title='Divorced Attrition')


# # Feature Engineering 

# In[73]:


# Encode categorical variables

encoder = LabelEncoder()
df["Attrition"] = encoder.fit_transform(df["Attrition"])
df = pd.get_dummies(df, drop_first=True)


# # Model Selection - RandomForest, Decision Trees, K-Nearest Neighbors (KNN)

# # Model training and evaluation:

# In[49]:


# Split the dataset into features and target
X = df.drop(columns=["Attrition"])
y = df["Attrition"]


# In[50]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


# Initialize and train the model
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_classifier.predict(X_test)

# Evaluate the model
print("Random Forest Classifier:")
print(classification_report(y_test, rf_predictions))


# In[52]:


# Initialize and train the model
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_classifier.predict(X_test)

# Evaluate the model
print("Decision Tree Classifier:")
print(classification_report(y_test, dt_predictions))


# In[53]:


# Initialize and train the model
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

# Make predictions
knn_predictions = knn_classifier.predict(X_test)

# Evaluate the model
print("K-Nearest Neighbors Classifier:")
print(classification_report(y_test, knn_predictions))


# # Hyperparameter tuning

# In[74]:


# Hyperparameter tuning using Random Search for all algorithms

# KNN classifier
param_dist_knn = {
    'n_neighbors': randint(1, 20),
    'weights': ['uniform', 'distance']
}
random_search_knn = RandomizedSearchCV(estimator=knn_classifier, param_distributions=param_dist_knn, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search_knn.fit(X_train, y_train)
print("Best Hyperparameters (KNN):", random_search_knn.best_params_)
print("Best Score (KNN):", random_search_knn.best_score_)

# Random Forest Classifier
param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}
random_search_rf = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search_rf.fit(X_train, y_train)
print("Best Hyperparameters (Random Forest):", random_search_rf.best_params_)
print("Best Score (Random Forest):", random_search_rf.best_score_)

# Decision Tree Classifier
param_dist_dt = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}
random_search_dt = RandomizedSearchCV(estimator=dt_classifier, param_distributions=param_dist_dt, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search_dt.fit(X_train, y_train)
print("Best Hyperparameters (Decision Tree):", random_search_dt.best_params_)
print("Best Score (Decision Tree):", random_search_dt.best_score_)


# In[82]:


# Evaluate tuned models on the test set

# KNN Classifier
knn_tuned = random_search_knn.best_estimator_
knn_pred = knn_tuned.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print("Accuracy (KNN) after hyperparameter tuning:", knn_accuracy)

# Random Forest Classifier
rf_tuned = random_search_rf.best_estimator_
rf_pred = rf_tuned.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Accuracy (Random Forest) after hyperparameter tuning:", rf_accuracy)

# Decision Tree Classifier
dt_tuned = random_search_dt.best_estimator_
dt_pred = dt_tuned.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Accuracy (Decision Tree) after hyperparameter tuning:", dt_accuracy)


# # Confusion Matrix to provide a comprehensive summary of the model's performance 

# In[77]:


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Decision Tree Confusion Matrix
dt_pred = random_search_dt.best_estimator_.predict(X_test)  # Replace with actual prediction
plot_confusion_matrix(y_test, dt_pred, title='Decision Tree Confusion Matrix')

# Random Forest Confusion Matrix
rf_pred = random_search_rf.best_estimator_.predict(X_test)  # Replace with actual prediction
plot_confusion_matrix(y_test, rf_pred, title='Random Forest Confusion Matrix')

# KNN Confusion Matrix
knn_pred = random_search_knn.best_estimator_.predict(X_test)  # Replace with actual prediction
plot_confusion_matrix(y_test, knn_pred, title='KNN Confusion Matrix')


# In[ ]:




