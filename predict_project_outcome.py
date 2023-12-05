import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Read dataset
df = pd.read_csv("projects.csv")
df.describe()


# One hot encode the columns country and domain
df_encoded = pd.get_dummies(df, columns=['country', 'domain'])
# Drop the column name
df = df_encoded.drop(['name'], axis=1)

df = df.sample(frac=1.0, random_state=42)

# Reset the index to maintain a consistent index for the shuffled data
df.reset_index(drop=True, inplace=True)




X = df.drop(['outcome'], axis=1)
y = df['outcome']

# Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# create logistic regression model
model = LogisticRegression()

# Fit the model with the training data
model.fit(X_train, y_train)

# Predict the test
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)


# Print the accuracy evaluation
print("Accuracy:", accuracy)

print("Confusion Matrix:\n", confusion)

print("Classification Report:\n", report)

