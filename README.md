# Machine-Learning-Assignments

Assignments for EE4C12: Machine Learning for Electrical Engineering Applications at TU Delft (2024).

The course covers ML workflows in Python using NumPy, Pandas, and scikit-learn, applied to real-world engineering datasets.

## Lab 1 – Python Fundamentals & Data Exploration
**Dataset:** Radar-based gesture recognition data (Click, Pinch, Swipe, Wave)

Covers NumPy array operations, descriptive statistics, histogram and boxplot visualisation, Pandas DataFrames, and implementing a custom stratified train-test split from scratch.

## Lab 2 – Regression & Energy Load Forecasting
**Dataset:** Hourly electricity consumption data with weather features

Implements linear regression, Ridge regression, and Lasso regression to forecast active power demand. Explores feature engineering with one-hot encoded time features, data standardisation, and hyperparameter tuning via grid search. Also includes support vector regression (SVR) with linear, polynomial, and RBF kernels.

## Lab 3 – Classification of Network Traffic
**Dataset:** Network traffic data with 6 features and 4 usage categories (High, Varied, Low, Buffer)

Implements binary and multiclass classification using logistic regression (one-vs-all and multinomial) and support vector machines (linear and polynomial kernels). Topics include ROC curves, confusion matrices, class weighting to reduce false negatives, and hyperparameter tuning of C and tolerance.

## Lab 4 – Medical Image Classification
**Dataset:** 400 medical images across 4 classes — Abdomen CT, Brain CT, normal Chest X-rays, Chest X-rays with Pneumonia

Covers image preprocessing (grayscale conversion, resizing), manual zero-padding and pooling, Sobel kernel convolution for gradient computation, Histogram of Oriented Gradients (HOG) feature extraction, and PCA-based dimensionality reduction. Classifiers (KNN, SVC, MLP) are compared across raw flattened images, HOG features, and PCA-transformed features.
