import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data-db.csv') #data-db.csv is the file containing all data

# Assuming your data is stored in a variable 'data'
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Labels

# Step 2: Prepare Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Models
rf_model = RandomForestClassifier()
svm_model = SVC()
lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)
nn_model = MLPClassifier(max_iter=1000)
dt_model = DecisionTreeClassifier(ccp_alpha=0.1)
xgb_model = XGBClassifier()
gb_model = GradientBoostingClassifier()

rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Step 4: Predict
rf_predictions = rf_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)
nn_predictions = nn_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Step 5: Get Accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)
nn_accuracy = accuracy_score(y_test, nn_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
gb_accuracy = accuracy_score(y_test, gb_predictions)

# Step 6: Compare Results
accuracies = {
    'Random Forest': rf_accuracy,
    'SVM': svm_accuracy,
    'Logistic Regression': lr_accuracy,
    'Neural Network': nn_accuracy,
    'Decision Tree': dt_accuracy,
    'XGB Classifier': xgb_accuracy,
    'Gradient Boost': gb_accuracy

}

best_model = max(accuracies, key=accuracies.get)

print("Model Accuracies:")
for model, accuracy in accuracies.items():
    print(f"{model}: {accuracy}")

print(f"\nThe best model is: {best_model}")
