# GitHub Copilot Review Instructions

## Basic Instructions
- **Always respond in Japanese (日本語)**
- Provide detailed and constructive feedback in code reviews
- Suggest specific and implementable improvements

## Code Quality Checklist

### General Coding Standards
- Check PEP8 compliance (for Python)
- Verify proper variable and function naming
- Review comments and documentation appropriateness
- Ensure proper error handling implementation

### Security and Best Practices
- Check for security vulnerabilities
- Suggest performance optimizations
- Review memory leaks and resource management

## Machine Learning Project Specific Requirements

When reviewing machine learning related code, focus on the following points:

### 1. MLflow Model Management
- [ ] **Verify MLflow model management implementation**
  - Check if `mlflow.start_run()` is properly used to start runs
  - Ensure `mlflow.log_param()` is used to log hyperparameters
  - Verify `mlflow.log_metric()` is used to log model evaluation metrics
  - Confirm `mlflow.end_run()` is properly called to end runs

### 2. MLflow Model Saving
- [ ] **Ensure all trained models are saved using MLflow**
  - Check if `mlflow.sklearn.log_model()` or appropriate framework-specific functions are used
  - Verify proper metadata (model name, version, description) is included when saving models
  - Ensure necessary information for reproducibility (dependencies, environment info) is recorded

### 3. Inference Result Visualization
- [ ] **Verify that inference results are properly plotted**
  - Prediction vs actual value scatter plots (for regression problems)
  - Confusion matrix and ROC curves (for classification problems)
  - Residual plots (for regression problems)
  - Feature importance visualization
  - Proper use of `matplotlib`, `seaborn`, `plotly` for appropriate visualization

### 4. Feature Engineering and Intermediate Table Saving
- [ ] **Check if intermediate tables are saved after feature generation**
  - Saving preprocessed data (CSV, Parquet, pickle, etc.)
  - Saving feature engineering results
  - Version control for data reproducibility
  - Recording data schema

### 5. sklearn Pipeline Utilization
- [ ] **Verify that ML workflows are built using sklearn Pipeline**
  - Use of `sklearn.pipeline.Pipeline` to integrate preprocessing and model training
  - Use of `sklearn.compose.ColumnTransformer` for column-wise preprocessing
  - Proper definition of each pipeline step
  - Ensure pipeline is reusable and maintainable

## Specific Review Checkpoints

### Data Processing
- Prevent data leakage (proper separation of training and validation data)
- Proper cross-validation implementation
- Appropriate missing value handling
- Outlier detection and processing

### Model Evaluation
- Appropriate evaluation metric selection
- Detection of overfitting/underfitting
- Comparison with baseline models
- Statistical significance validation

### Implementation
- Code reproducibility assurance
- Random seed fixing
- Execution environment recording
- Clear dependency management

## Feedback Format

Provide review comments in the following format:

```
**Issue**: [Description of the problem]
**Improvement Suggestion**: [Specific improvement method]
**Reason**: [Why this improvement is necessary]
**Sample Code**: [Provide improvement example if possible]
```

## Exceptions

Apply the above requirements partially in the following cases:
- Prototype or experimental code
- Unit tests
- EDA (Exploratory Data Analysis) only code

---

*This file is designed to improve code quality in machine learning projects. Please adjust as needed based on project progress and requirements.*