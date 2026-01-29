import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
data_path = '../data/titanic/train.csv'
df = pd.read_csv(data_path)

# 2. Preprocessing
# Drop columns that are difficult to use or irrelevant for this simple baseline
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Define features
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Split Data
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define 5 Individual Models
clf1 = LogisticRegression(random_state=1)
clf2 = DecisionTreeClassifier(random_state=1)
clf3 = RandomForestClassifier(random_state=1)
clf4 = SVC(probability=True, random_state=1) # Probability=True needed for soft voting if used
clf5 = KNeighborsClassifier()

# 5. Create Ensemble Model (VotingClassifier)
# We will use 'soft' voting for probability averaging, or 'hard' for class label voting. 
# 'hard' is robust and simple.
eclf = VotingClassifier(
    estimators=[
        ('lr', clf1), 
        ('dt', clf2), 
        ('rf', clf3), 
        ('svc', clf4), 
        ('knn', clf5)
    ],
    voting='hard'
)

# Create full pipeline including preprocessing
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', eclf)])

# 6. Train and Evaluate
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Voting Ensemble Accuracy: {accuracy:.4f}')

# 7. Save Model
joblib.dump(model_pipeline, 'titanic_voting_model.pkl')
print("Model saved as 'titanic_voting_model.pkl'")
