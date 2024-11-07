# Principal Component Analysis - Handwritten Digit Recognition
This project demonstrates the use of Principal Component Analysis (PCA) for dimensionality reduction and classification of handwritten digits from the Digits Dataset. The dataset consists of 1797 8x8 pixel images of handwritten digits from 0 to 9. The goal of the project is to perform PCA to reduce the dimensionality of the dataset and use the reduced data for classification with a K-Nearest Neighbors (KNN) classifier.

Libraries Used
numpy: For numerical operations.
matplotlib: For plotting and visualizations.
scikit-learn: For machine learning algorithms, data preprocessing, and PCA implementation.
StandardScaler: For data normalization.
KNeighborsClassifier: For the classification task.
Features
Data Preprocessing:

The dataset is split into a training set (80%) and a test set (20%).
Standard scaling is applied to normalize the data before applying PCA.
Principal Component Analysis (PCA):

PCA is performed to reduce the dimensionality of the dataset.
A graph is plotted showing the cumulative variance explained by the principal components.
Classification:

K-Nearest Neighbors (KNN) is used to classify the reduced data.
The classifier is trained on the training set and evaluated on the test set.
Visualization:

A gallery of test images is displayed along with their true and predicted labels to visualize the model’s performance.