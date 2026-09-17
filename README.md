# TItanic_Keggle_ML
An end-to-end machine learning pipeline built to predict passenger survival on the Titanic (Kaggle competition). This repository demonstrates the progression from building a custom classifier from scratch to deploying a fully optimized tree-based model.
Project Highlights
Custom Logistic Regression: Implemented entirely from scratch using NumPy, featuring a custom training loop with gradient descent and a sigmoid activation function.

Deep Neural Network: Built with TensorFlow/Keras, utilizing 4 hidden layers and Leaky ReLU activations to capture non-linear relationships in the passenger data.

Optimized Random Forest: The final production model, tuned using GridSearchCV with 5-Fold Cross-Validation to identify the best hyperparameters and prevent overfitting.

Data Processing Pipeline
The dataset undergoes strict preprocessing to prevent data leakage and handle class imbalance:

Imputation: Missing Age and Fare values are filled using the training set's mean (improving performance over median imputation).

Encoding & Scaling: Categorical variables are mapped to binary, and all features are standardized.

SMOTE (Synthetic Minority Over-sampling Technique): Applied exclusively to the training data to balance the survival classes.

Note: Principal Component Analysis (PCA) was initially tested for dimensionality reduction but was ultimately removed, as allowing the models to train directly on the raw, uncompressed features yielded higher accuracy.

Repository Structure
train.csv / test.csv: The raw Kaggle datasets.

pipeline.py: Contains the data preprocessing steps, the custom Logistic Regression class, the Neural Network architecture, and the Random Forest Grid Search.

submission_*.csv: The generated prediction files formatted and ready for Kaggle scoring.
