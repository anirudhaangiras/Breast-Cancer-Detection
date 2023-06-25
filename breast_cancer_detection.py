# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

# Importing the Data
cancer_data = pd.read_csv("breast_cancer.csv")

# Separate Features
X = cancer_data.iloc[:,1:-1].values
y = cancer_data.iloc[:,-1].values

# Convert the Target Variable to Binary Values
y = np.where(y == 2, 0, 1)

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

# Feature scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Creating an Instance of Logistic Regression
model = LogisticRegression(random_state = 0)

# Performing Hyperparameter Tuning using Grid Search
parameters = { 'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(estimator = model, param_grid = parameters, scoring = 'accuracy', cv=10, n_jobs = -1)
grid_search.fit(X_train_scaled, y_train)
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_
print("Best Accuracy: {:.2f}%".format(best_accuracy*100))
print("Best Parameters:", best_params)

# Training the Model
best_model = LogisticRegression(**best_params)
best_model.fit(X_train_scaled, y_train)

# Testing the Model
y_pred = best_model.predict(X_test_scaled)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Creating the Confusion Matrix and Accuracy Score
print(accuracy_score(y_test, y_pred),"\n")
mat = confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(mat, annot=True, linewidth=0.7, linecolor='red', fmt='g', ax=ax, cmap="BuPu")
plt.title('Logistic Regression Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

# Accuracy with K-Fold Cross Validation
accuracy = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f}%".format(accuracy.mean()*100))
print("Standard Deviation: {:.2f}%".format(accuracy.std()*100))

# Generate an ROC curve and calculating the AUC
pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, pred_prob)
auc = roc_auc_score(y_test, pred_prob)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
