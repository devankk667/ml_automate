import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def load_iris_data():
    """Load and prepare the Iris dataset."""
    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # Add some random noise to make it more interesting
    np.random.seed(42)
    for col in X.columns:
        X[col] = X[col] + np.random.normal(0, 0.1, size=len(X))
    
    # Add some missing values
    for col in X.columns:
        mask = np.random.random(len(X)) < 0.05  # 5% missing values
        X.loc[mask, col] = np.nan
    
    return X, y, iris.target_names

def main():
    # Load and prepare data
    X, y, target_names = load_iris_data()
    
    # Handle missing values (simple imputation with mean)
    X = X.fillna(X.mean())
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize and train a simple Random Forest classifier
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest set accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save the model
    model_path = 'output/iris_rf_model.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to '{model_path}'")
    
    # Plot feature importances
    plot_feature_importance(model, X.columns)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, target_names)
    
    # Create pair plot of the features
    plot_pair_plot(pd.concat([X, y], axis=1), 'target')
    
    # Plot decision boundaries (using first two principal components)
    plot_decision_boundary_2d(model, X, y, target_names)

def plot_feature_importance(model, feature_names):
    """Plot feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/feature_importance.png', bbox_inches='tight')
    plt.close()
    print("\nSaved feature importance plot to 'output/plots/feature_importance.png'")

def plot_confusion_matrix(y_true, y_pred, target_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    
    # Save the plot
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/confusion_matrix.png', bbox_inches='tight')
    plt.close()
    print("Saved confusion matrix to 'output/plots/confusion_matrix.png'")

def plot_pair_plot(data, target_col):
    """Create a pair plot of the features."""
    # Use a subset of data for better visualization
    plot_data = data.sample(min(100, len(data)), random_state=42)
    
    # Create pair plot
    pair_plot = sns.pairplot(plot_data, hue=target_col, palette='viridis', 
                            plot_kws={'alpha': 0.8, 's': 50, 'edgecolor': 'k'})
    
    # Adjust layout
    plt.suptitle('Pair Plot of Iris Features', y=1.02)
    
    # Save the plot
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/pair_plot.png', bbox_inches='tight')
    plt.close()
    print("Saved pair plot to 'output/plots/pair_plot.png'")

def plot_decision_boundary_2d(model, X, y, target_names):
    """Plot decision boundaries using first two principal components."""
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    # Create a mesh grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Train a new model on the PCA-transformed data
    model_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    model_pca.fit(X_pca, y)
    
    # Predict on the mesh grid
    Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#00AA00', '#0000FF']
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    
    # Plot the training points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=ListedColormap(cmap_bold),
                         edgecolor='k', s=50, alpha=0.8)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary (PCA-reduced data)")
    plt.xlabel(f"First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})")
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=name,
                                markerfacecolor=color, markersize=10, markeredgecolor='k')
                      for name, color in zip(target_names, cmap_bold)]
    plt.legend(handles=legend_elements, title="Classes")
    
    # Save the plot
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/decision_boundary.png', bbox_inches='tight')
    plt.close()
    print("Saved decision boundary plot to 'output/plots/decision_boundary.png'")

if __name__ == "__main__":
    main()
