import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Configuration
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)
np.random.seed(42)
sns.set_style("whitegrid")

def load_and_preprocess():
    print("1. Loading and preprocessing data...")
    df = pd.read_csv('House_Prices.csv')

    print(f"\nDataset shape: {df.shape}")
    print("\nData sample:")
    print(df.head(3))
    print("\nDescription:\n", df.describe())
    print("\nMissing values:\n", df.isnull().sum())

    # Convert yes/no to binary
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))
    
    # Encode furnishing status
    le = LabelEncoder()
    df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])
    
    return df

def engineer_features(df):
    print("\n2. Engineering features...")
    # Basic features
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['area_per_room'] = df['area'] / (df['total_rooms'] + 0.1)
    df['has_parking'] = (df['parking'] > 0).astype(int)
    
    # Interaction features
    df['stories_area'] = df['stories'] * df['area']
    df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.1)
    
    # Composite features
    df['luxury_score'] = (df['airconditioning'] + df['prefarea'] + 
                         (df['furnishingstatus'] == 2).astype(int) + 
                         df['has_parking'])
    
    df['amenities_score'] = (df['guestroom'] + df['basement'] + 
                            df['hotwaterheating'] + df['mainroad'])
    
    return df

def analyze_features(df):
    print("\n3. Analyzing features...")
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr[abs(corr['price']) > 0.3], annot=True, cmap='coolwarm', center=0)
    plt.title("Features Correlating with Price (|r| > 0.3)")
    plt.tight_layout()
    plt.show()

def evaluate_model(X, y, features, name):
    X = X[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=150, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{name} Model (n_features={len(features)})")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Get feature importances
    importances = model.named_steps['regressor'].feature_importances_
    feat_importances = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 Features:")
    print(feat_importances.head(5))
    
    # Plot top 10 features
    plt.figure(figsize=(10, 6))
    top_features = feat_importances.head(10)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'{name} Top Feature Importances')
    plt.tight_layout()
    plt.show()
    
    return rmse, r2, feat_importances

def main():
    # Load and preprocess
    df = load_and_preprocess()
    
    # Define original features
    original_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                        'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                        'parking', 'prefarea', 'furnishingstatus']
    
    # Engineer new features
    df = engineer_features(df)
    engineered_features = original_features + [
        'total_rooms', 'area_per_room', 'luxury_score', 'has_parking',
        'stories_area', 'bed_bath_ratio', 'amenities_score'
    ]
    
    # Analyze features
    analyze_features(df)
    
    # Prepare data
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Evaluate models
    print("\n4. Evaluating models...")
    orig_rmse, orig_r2, _ = evaluate_model(X, y, original_features, "Original")
    all_rmse, all_r2, _ = evaluate_model(X, y, engineered_features, "Engineered")
    
    # Feature selection
    print("\n5. Feature selection...")
    selector = SelectKBest(f_regression, k=12)
    X_engineered = X[engineered_features]
    selector.fit(X_engineered, y)
    selected_mask = selector.get_support()
    selected_features = [f for f, m in zip(engineered_features, selected_mask) if m]
    sel_rmse, sel_r2, _ = evaluate_model(X, y, selected_features, "Selected")
    
    # Results comparison
    print("\n6. Final comparison:")
    results = pd.DataFrame({
        'Feature Set': ['Original', 'Engineered', 'Selected'],
        'Features': [len(original_features), len(engineered_features), len(selected_features)],
        'RMSE': [orig_rmse, all_rmse, sel_rmse],
        'R2': [orig_r2, all_r2, sel_r2]
    })
    print(results.to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Feature Set', y='R2', data=results)
    plt.title('Model Performance Comparison')
    plt.ylabel('R2 Score')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    print("\nScript completed successfully!")