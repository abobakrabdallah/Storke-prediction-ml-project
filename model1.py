import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
# %matplotlib inline
import warnings
warnings.filterwarnings(action='ignore')
data = pd.read_csv('C:/Desktop/healthcare-dataset-stroke-data.csv')
print(data.head(5))
print(data.shape)
print(data.info())
print(data.isnull().sum())

print(data.columns)
# Fill missing values only in numeric columns
data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)
print(data.head(5))

# # # # # # # # # # # Outlier Analysis
# Calculate Q1 and Q3 only on numeric columns
Q1 = data.select_dtypes(include=[np.number]).quantile(0.25)
Q3 = data.select_dtypes(include=[np.number]).quantile(0.75)
# Calculate the interquartile range for numeric columns
IQR = Q3 - Q1

# Filter out outliers only in numeric columns, keeping non-numeric columns as they are
numeric_data = data.select_dtypes(include=[np.number])
non_numeric_data = data.select_dtypes(exclude=[np.number])

# Apply outlier filter only to numeric columns
numeric_data = numeric_data[~((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Concatenate the filtered numeric data with the non-numeric data
data = pd.concat([numeric_data, non_numeric_data], axis=1)

print(data.describe())
print(data.head(5))
# ___________________________________________________________________________________________________

# # # # # # # # # # # Binning
data['age_binned'] = pd.cut(data['age'], bins=[0, 30, 60, 100], labels=['Young', 'Middle-aged', 'Old'])

print(data[['age', 'age_binned']].head())

# ___________________________________________________________________________________________________
# # # # # # # # # # #  PCA
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

X = data.drop('stroke', axis=1)
y = data['stroke']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

#PCA result
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=40)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Stroke Dataset')
plt.colorbar(label='Stroke')
plt.show()


# ___________________________________________________________________________________________________
# # # # # # # # # # #  
relevant_columns = ['age', 'avg_glucose_level', 'bmi']

for column in relevant_columns:
    mean = data[column].mean()           
    median = data[column].median()     
    mode = data[column].mode()[0]        
    std_dev = data[column].std()        
    variance = data[column].var()    
        

    print(f"Statistics for {column}:")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")
    print("----------")



# for column in relevant_columns:
plt.figure(figsize=(8, 5))
sns.histplot(data[column], kde=True)
plt.title(f"Distribution of {column}")
plt.xlabel(column)
plt.ylabel("Frequency")
plt.show()


# ___________________________________________________________________________________________________
# # # # # # # # # # # Regression
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, x.shape)  


data = pd.DataFrame({'x': x, 'y': y})


X_train, X_test, y_train, y_test = train_test_split(data[['x']], data['y'], test_size=0.2, random_state=42)


degree = 4
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X_train)


model = LinearRegression()
model.fit(X_poly, y_train)

X_test_poly = poly_features.transform(X_test)
y_pred = model.predict(X_test_poly)

plt.scatter(data['x'], data['y'], color='lightgray', label='Data points')
plt.scatter(X_test, y_test, color='red', label='Test data', marker='x')
plt.scatter(X_test, y_pred, color='blue', label='Predictions', marker='o')
plt.title('Polynomial Regression for Data Smoothing')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()






# ___________________________________________________________________________________________________
# # # # # # # # # # #  correlation matrix
data['bmi'] = pd.to_numeric(data['bmi'].replace('N/A', pd.NA))

data['bmi'] = data['bmi'].fillna(data['bmi'].median())


categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)


correlation_matrix = data_encoded.corr()


plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, 
            annot=True,           
            cmap='coolwarm',     
            fmt='.2f',            
            square=True,         
            mask=np.triu(correlation_matrix)) 

plt.title('Correlation Matrix for Stroke Prediction Dataset')
plt.tight_layout()
plt.show()


print("\nTop correlations with stroke:")
stroke_correlations = correlation_matrix['stroke'].sort_values(ascending=False)
print(stroke_correlations)












# ___________________________________________________________________________________________________
# # # # # # # # # # #  


