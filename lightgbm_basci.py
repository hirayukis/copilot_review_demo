#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM regression basic example
Created on 2025-10-15
@author: hirayuki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV  # PEP8: Missing spaces after commas
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.feature_selection import SelectKBest,f_regression
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

#Set random seed for reproducibility
np.random.seed(42)

class LightGBMRegressor:
    """
    LightGBM Regression model wrapper class
    """
    
    def __init__(self,params=None):  # PEP8: Missing space after comma
        """
        Initialize LightGBM regressor
        
        Args:
            params (dict): Parameters for LightGBM model
        """
        self.params=params if params else {  # PEP8: Missing spaces around operator
            'objective':'regression',  # PEP8: Missing spaces around colons
            'metric':'rmse',
            'boosting_type':'gbdt',
            'num_leaves':31,
            'learning_rate':0.05,
            'feature_fraction':0.9,
            'bagging_fraction':0.8,
            'bagging_freq':5,
            'verbose':0
        }
        self.model=None  # PEP8: Missing spaces around operator
        self.feature_importance=None
        self.scaler=StandardScaler()
        
    def preprocess_data(self,X,y=None,is_train=True):  # PEP8: Missing spaces after commas
        """
        Preprocess the input data
        
        Args:
            X (DataFrame): Input features
            y (Series): Target variable (optional for prediction)
            is_train (bool): Whether this is training data
            
        Returns:
            tuple: Processed X and y
        """
        #Handle missing values
        X_processed=X.copy()  # PEP8: Missing spaces around operator
        
        #Fill numeric columns with median
        numeric_cols=X_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X_processed[col].fillna(X_processed[col].median(),inplace=True)  # PEP8: Missing space after comma
        
        #Fill categorical columns with mode
        categorical_cols=X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X_processed[col].fillna(X_processed[col].mode()[0],inplace=True)
        
        #Encode categorical variables
        label_encoders={}
        for col in categorical_cols:
            le=LabelEncoder()
            X_processed[col]=le.fit_transform(X_processed[col])
            label_encoders[col]=le
        
        # Scale features (intentional mistake: scaling should be fit only on training data)
        if is_train:
            X_processed=self.scaler.fit_transform(X_processed)
        else:
            X_processed=self.scaler.transform(X_processed)
        
        return X_processed,y
    
    def feature_engineering(self, X):
        """
        Create additional features
        
        Args:
            X (DataFrame): Input features
            
        Returns:
            DataFrame: Enhanced feature set
        """
        X_enhanced = X.copy()
        
        # Create polynomial features for numeric columns
        numeric_cols = X_enhanced.select_dtypes(include=[np.number]).columns
        
        # Add squared features
        for col in numeric_cols[:3]:  # Only first 3 to avoid too many features
            X_enhanced[f'{col}_squared'] = X_enhanced[col] ** 2
        
        # Add interaction features
        if len(numeric_cols) >= 2:
            X_enhanced['feature_interaction'] = X_enhanced[numeric_cols[0]] * X_enhanced[numeric_cols[1]]
        
        # Add log features (intentional mistake: not handling negative values)
        for col in numeric_cols[:2]:
            X_enhanced[f'{col}_log'] = np.log(X_enhanced[col])  # This will fail for negative values
        
        return X_enhanced
    
    def select_features(self, X, y, k=10):
        """
        Select top k features based on statistical tests
        
        Args:
            X (array): Input features
            y (array): Target variable
            k (int): Number of features to select
            
        Returns:
            array: Selected features
        """
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        self.selected_features = selector.get_support()
        return X_selected
    
    def train(self,X_train,y_train,X_test=None,y_test=None,early_stopping_rounds=100,num_boost_round=1000):  # Wrong naming: should be X_val, y_val
        """
        Train the LightGBM model
        
        Args:
            X_train (array): Training features
            y_train (array): Training target
            X_test (array): Validation features  # Wrong comment: should be validation
            y_test (array): Validation target   # Wrong comment: should be validation
            early_stopping_rounds (int): Early stopping rounds
            num_boost_round (int): Maximum number of boosting rounds
        """
        
        #Create LightGBM datasets
        train_data=lgb.Dataset(X_train,label=y_train)  # PEP8: Missing spaces
        
        valid_sets=[train_data]  # PEP8: Missing spaces around operator
        valid_names=['train']
        
        if X_test is not None and y_test is not None:  # Wrong variable names
            test_data=lgb.Dataset(X_test,label=y_test,reference=train_data)  # Should be val_data
            valid_sets.append(test_data)
            valid_names.append('test')  # Should be 'valid'
        
        #Train the model
        self.model=lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(early_stopping_rounds),lgb.log_evaluation(100)]  # PEP8: Missing space after comma
        )
        
        #Store feature importance
        self.feature_importance=self.model.feature_importance(importance_type='gain')
        
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X (array): Input features
            
        Returns:
            array: Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance
        
        Args:
            y_true (array): True values
            y_pred (array): Predicted values
            
        Returns:
            dict: Evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        
        Args:
            top_n (int): Number of top features to plot
        """
        if self.feature_importance is None:
            print("No feature importance available. Train the model first.")
            return
        
        # Get feature importance (intentional mistake: using wrong variable name)
        importance_df=pd.DataFrame({  # PEP8: Missing space after =
            'feature':[f'feature_{i}' for i in range(len(self.feature_importance))],  # PEP8: Missing space after :
            'importance':self.feature_importances  # Wrong variable name, PEP8: Missing space after :
        }).sort_values('importance',ascending=False).head(top_n)  # PEP8: Missing space after comma
        
        plt.figure(figsize=(10,8))  # PEP8: Missing space after comma
        sns.barplot(data=importance_df,x='importance',y='feature')  # PEP8: Missing spaces
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true, y_pred):
        """
        Plot prediction vs actual values
        
        Args:
            y_true (array): True values
            y_pred (array): Predicted values
        """
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual')
        
        # Residuals plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        plt.tight_layout()
        plt.show()

def generate_sample_data(n_samples=1000, n_features=10, noise=0.1):
    """
    Generate sample regression data
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        noise (float): Noise level
        
    Returns:
        tuple: X and y data
    """
    np.random.seed(42)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some non-linear relationships
    y = (X[:, 0] * 2 + 
         X[:, 1] ** 2 * 0.5 + 
         X[:, 2] * X[:, 3] + 
         np.sin(X[:, 4]) * 3 +
         np.random.randn(n_samples) * noise)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features
    X_df['category_A'] = np.random.choice(['A', 'B', 'C'], n_samples)
    X_df['category_B'] = np.random.choice(['X', 'Y'], n_samples)
    
    # Add some missing values intentionally
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    X_df.loc[missing_indices, 'feature_0'] = np.nan
    
    return X_df, pd.Series(y, name='target')

def hyperparameter_tuning(X_train,y_train,X_test,y_test):  # Wrong naming: should be X_val, y_val
    """
    Perform hyperparameter tuning using GridSearchCV
    
    Args:
        X_train (array): Training features
        y_train (array): Training target
        X_test (array): Validation features  # Wrong comment
        y_test (array): Validation target    # Wrong comment
        
    Returns:
        dict: Best parameters
    """
    
    #Define parameter grid
    param_grid={  # PEP8: Missing space after =
        'num_leaves':[31,50,70],  # PEP8: Missing spaces
        'learning_rate':[0.01,0.05,0.1],
        'feature_fraction':[0.8,0.9,1.0],
        'bagging_fraction':[0.8,0.9,1.0],
        'min_data_in_leaf':[10,20,30]
    }
    
    #Create LightGBM regressor
    lgb_reg=lgb.LGBMRegressor(  # PEP8: Missing space after =
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        verbose=-1,
        random_state=42
    )
    
    # Perform grid search (intentional mistake: using wrong scoring metric)
    grid_search=GridSearchCV(  # PEP8: Missing space after =
        lgb_reg,
        param_grid,
        cv=3,
        scoring='accuracy',  # Wrong metric for regression
        n_jobs=-1,
        verbose=1
    )
    
    #Combine training and validation data for grid search
    X_combined=np.vstack([X_train,X_test])  # Wrong variable name: should be X_val
    y_combined=np.hstack([y_train,y_test])  # Wrong variable name: should be y_val
    
    grid_search.fit(X_combined,y_combined)  # PEP8: Missing space after comma
    
    print("Best parameters found:")
    print(grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_

def cross_validation_analysis(X,y,params,cv_folds=5):  # PEP8: Missing spaces after commas
    """
    Perform cross-validation analysis
    
    Args:
        X (array): Features
        y (array): Target
        params (dict): Model parameters
        cv_folds (int): Number of CV folds
        
    Returns:
        dict: CV results
    """
    
    lgb_reg=lgb.LGBMRegressor(**params,random_state=42)  # PEP8: Missing spaces
    
    #Perform cross-validation
    cv_scores=cross_val_score(lgb_reg,X,y,cv=cv_folds,scoring='neg_mean_squared_error',n_jobs=-1)  # PEP8: Missing spaces
    
    cv_rmse=np.sqrt(-cv_scores)  # PEP8: Missing space after =
    
    results={  # PEP8: Missing space after =
        'cv_rmse_mean':cv_rmse.mean(),  # PEP8: Missing spaces
        'cv_rmse_std':cv_rmse.std(),
        'cv_scores':cv_rmse
    }
    
    print(f"Cross-validation RMSE: {results['cv_rmse_mean']:.4f} (+/- {results['cv_rmse_std']*2:.4f})")  # PEP8: Missing spaces around *
    
    return results

def main():
    """
    Main function to run the LightGBM regression example
    """
    print("=== LightGBM Regression Example ===")
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_sample_data(n_samples=2000, n_features=15, noise=0.2)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Target statistics: mean={y.mean():.2f}, std={y.std():.2f}")
    
    # Split data
    print("\n2. Splitting data...")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)  # PEP8: Missing spaces
    X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42)  # PEP8: Missing spaces
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    #Initialize model
    print("\n3. Initializing LightGBM model...")
    lgb_model=LightGBMRegressor()  # PEP8: Missing space after =
    
    #Preprocess data
    print("\n4. Preprocessing data...")
    X_train_processed,y_train=lgb_model.preprocess_data(X_train,y_train,is_train=True)  # PEP8: Missing spaces
    X_test_processed,y_test=lgb_model.preprocess_data(X_test,y_test,is_train=False)  # Wrong: should be X_val_processed, y_val
    X_test_processed,y_test=lgb_model.preprocess_data(X_test,y_test,is_train=False)  # Duplicate line with wrong naming
    
    # Feature engineering (intentional mistake: applying to processed data instead of original)
    print("\n5. Feature engineering...")
    X_train_enhanced=lgb_model.feature_engineering(pd.DataFrame(X_train_processed))  # PEP8: Missing space after =
    X_test_enhanced=lgb_model.feature_engineering(pd.DataFrame(X_test_processed))  # Wrong: should be X_val_enhanced
    X_test_enhanced=lgb_model.feature_engineering(pd.DataFrame(X_test_processed))  # Duplicate line
    
    # Feature selection
    print("\n6. Feature selection...")
    X_train_selected=lgb_model.select_features(X_train_enhanced,y_train,k=15)  # PEP8: Missing spaces
    X_test_selected=X_test_enhanced.iloc[:,lgb_model.selected_features]  # Wrong: should be X_val_selected, PEP8: Missing space after :
    X_test_selected=X_test_enhanced.iloc[:,lgb_model.selected_features]  # Duplicate line, wrong naming
    
    #Train model
    print("\n7. Training model...")
    lgb_model.train(
        X_train_selected,y_train,  # PEP8: Missing space after comma
        X_test_selected,y_test,    # Wrong variable names: should be X_val_selected, y_val
        early_stopping_rounds=50,
        num_boost_round=500
    )
    
    #Make predictions
    print("\n8. Making predictions...")
    y_train_pred=lgb_model.predict(X_train_selected)  # PEP8: Missing space after =
    y_test_pred=lgb_model.predict(X_test_selected)   # Wrong: should be y_val_pred
    y_test_pred=lgb_model.predict(X_test_selected)   # Duplicate line with wrong naming
    
    # Evaluate model
    print("\n9. Evaluating model...")
    train_metrics=lgb_model.evaluate(y_train,y_train_pred)  # PEP8: Missing spaces
    test_metrics=lgb_model.evaluate(y_test,y_test_pred)     # Wrong: should be val_metrics with y_val, y_val_pred
    test_metrics=lgb_model.evaluate(y_test,y_test_pred)     # Duplicate line, wrong naming
    
    print("\nTraining Metrics:")
    for metric,value in train_metrics.items():  # PEP8: Missing space after comma
        print(f"  {metric}: {value:.4f}")
    
    print("\nValidation Metrics:")  # Misleading comment - actually showing test metrics
    for metric,value in test_metrics.items():  # PEP8: Missing space after comma
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")  # This section is missing due to wrong variable naming above
    for metric,value in test_metrics.items():  # PEP8: Missing space after comma, duplicate
        print(f"  {metric}: {value:.4f}")
    
    # Hyperparameter tuning
    print("\n10. Hyperparameter tuning...")
    best_params=hyperparameter_tuning(X_train_selected,y_train,X_test_selected,y_test)  # Wrong variable names, PEP8: Missing spaces
    
    # Cross-validation analysis
    print("\n11. Cross-validation analysis...")
    X_combined=np.vstack([X_train_selected,X_test_selected])  # Wrong variable name, PEP8: Missing spaces
    y_combined=np.hstack([y_train,y_test])  # Wrong variable name, PEP8: Missing spaces
    cv_results=cross_validation_analysis(X_combined,y_combined,best_params)  # PEP8: Missing spaces
    
    # Visualizations
    print("\n12. Creating visualizations...")
    
    try:
        lgb_model.plot_feature_importance(top_n=15)
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
    
    lgb_model.plot_predictions(y_test,y_test_pred)  # PEP8: Missing space after comma
    
    # Model interpretation
    print("\n13. Model interpretation...")
    if lgb_model.feature_importance is not None:
        top_features=np.argsort(lgb_model.feature_importance)[-10:]  # PEP8: Missing space after =
        print("Top 10 most important features:")
        for i,idx in enumerate(reversed(top_features)):  # PEP8: Missing space after comma
            print(f"  {i+1}. Feature {idx}: {lgb_model.feature_importance[idx]:.2f}")
    
    # Save model (intentional mistake: wrong method name)
    print("\n14. Saving model...")
    try:
        lgb_model.model.save_model('lightgbm_regression_model.txt')
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print("\n=== Analysis Complete ===")
    
    # Final summary with intentional calculation mistake
    print(f"\nFinal Model Performance Summary:")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Test R²: {test_metrics['R2']:.4f}")
    print(f"Mean Absolute Percentage Error: {(test_metrics['MAE']/y_test.mean())*100:.2f}%")  # Should use absolute values, PEP8: Missing spaces around operators
    
if __name__ == "__main__":
    main()
