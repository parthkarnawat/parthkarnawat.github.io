const n=`
# Building Credit Risk Models with Python

## Introduction

In this blog post, I'll share my experience developing IFRS9-compliant credit risk models using Python, specifically focusing on PD (Probability of Default) models.

## Key Components of a PD Model

1. **Data Preparation**
   - Historical default data
   - Macroeconomic variables
   - Customer attributes

2. **Feature Engineering**
   - Creating relevant ratios
   - Transformations for better model fit
   - Handling missing values

3. **Model Selection**
   - Logistic regression is often preferred for interpretability
   - Random forests and gradient boosting for non-linear relationships
   - Neural networks for complex patterns

4. **Model Validation**
   - Discrimination power (AUC, KS statistics)
   - Calibration (expected vs observed default rates)
   - Stability testing

## Python Implementation Example

\`\`\`python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load and prepare data
data = pd.read_csv('loan_data.csv')
features = ['loan_amount', 'income', 'debt_ratio', 'credit_history']
X = data[features]
y = data['default_flag']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc:.4f}")
\`\`\`

## IFRS9 Compliance Considerations

When developing PD models for IFRS9:
- Models must be forward-looking
- Need to incorporate macroeconomic scenarios
- Lifetime PD estimation is required
- Must account for point-in-time adjustments

## Conclusion

Python has become the preferred tool for risk modeling due to its flexibility, rich ecosystem of libraries, and reproducibility. As financial regulations evolve, the ability to quickly adapt models becomes increasingly important.
`;export{n as default};
