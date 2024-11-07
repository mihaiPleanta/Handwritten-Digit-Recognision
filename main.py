import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the MNIST digits dataset
digits = datasets.load_digits()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# Normalize the data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Apply PCA for dimensionality reduction on normalized data
n_components = 50  # You can adjust this parameter
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_normalized)
X_test_pca = pca.transform(X_test_normalized)

# Scree plot
eig_values = pca.explained_variance_
eig_values_cumsum = np.cumsum(eig_values)

# Plot the cumulative sum percentage
plt.plot(range(1, n_components + 1), eig_values_cumsum / eig_values_cumsum[-1])
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Sum Percentage')
plt.title('Scree Plot for PCA')
plt.show()

# Train a simple classifier (K-nearest neighbors in this example)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_pca)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualization (optional)
def plot_gallery(images, titles, h, w, n_row=3, n_col=6):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray_r)  # Use reversed grayscale colormap
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

n_row = 3
n_col = 6
sample_images = X_test[:n_row * n_col]
true_labels = y_test[:n_row * n_col]
predicted_labels = y_pred[:n_row * n_col]
titles = [f'True: {true}\nPredicted: {pred}' for true, pred in zip(true_labels, predicted_labels)]

plot_gallery(sample_images, titles, 8, 8)
plt.show()
