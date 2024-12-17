import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('processed_data.csv')

# Handle missing values
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())  # Replace missing BMI values with mean

# Encode categorical variables
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Define features (X) and target (y)
X = data.drop(['stroke'], axis=1)  # Drop the target column
y = data['stroke']  # Target column

# Combine X and y for balancing
data_combined = pd.concat([X, y], axis=1)

# Separate majority and minority classes
majority = data_combined[data_combined['stroke'] == 0]
minority = data_combined[data_combined['stroke'] == 1]

# Oversample the minority class
minority_oversampled = resample(
    minority,
    replace=True,                # Sample with replacement
    n_samples=len(majority),     # Match number of majority samples
    random_state=42              # For reproducibility
)

# Combine the oversampled minority class with the majority class
balanced_data = pd.concat([majority, minority_oversampled])

# Separate features and target again
X_balanced = balanced_data.drop(['stroke'], axis=1)
y_balanced = balanced_data['stroke']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5, random_state=42)  # You can adjust max_depth
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))

# Use the plot_tree function with class_names to display class labels
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["No Stroke", "Stroke"],
    filled=True,
    proportion=False,  # Do not show proportions
    impurity=True,     # Display Gini impurity
)

plt.show()
