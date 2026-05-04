import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(page_title="Vibe Coding: ML Visualizer", layout="wide")

st.title("🚀 Machine Learning Interactive Visualizer")
st.markdown("---")

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", [
    "1. Least Squares Regression",
    "2. KNN Comparison (Image Data)",
    "3. Linear Classifier Weights",
    "4. Gradient Descent (SGD vs Momentum)"
])

# --- Module 1: Least Squares Linear Regression ---
if menu == "1. Least Squares Regression":
    st.header("📈 Least Squares Linear Regression")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        n_points = st.slider("Number of Points", 10, 100, 50)
        noise = st.slider("Noise Level", 0.0, 5.0, 1.0)
    
    # Generate Synthetic Data
    np.random.seed(42)
    X = np.linspace(0, 10, n_points).reshape(-1, 1)
    y = 2.5 * X + 1 + np.random.normal(0, noise, (n_points, 1))
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X, y, color='#3498db', label='Actual Data', alpha=0.7)
    ax.plot(X, y_pred, color='#e74c3c', linewidth=2, label='Fitted Line (OLS)')
    ax.set_xlabel("Feature X")
    ax.set_ylabel("Target Y")
    ax.set_title("Linear Regression Visualization")
    ax.legend()
    st.pyplot(fig)
    
    st.success(f"**Model Equation:** $y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}$")

# --- Module 2: KNN Image Classification ---
elif menu == "2. KNN Comparison (Image Data)":
    st.header("🖼️ KNN Performance with different K values")
    st.info("Using Digits dataset (8x8 grayscale images).")
    
    # Load Data
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
    
    k_list = st.multiselect("Select K values to compare:", [1, 3, 5, 11, 21, 51], default=[1, 5, 51])
    
    if k_list:
        cols = st.columns(len(k_list))
        for i, k in enumerate(sorted(k_list)):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            acc = knn.score(X_test, y_test)
            
            with cols[i]:
                st.metric(f"K = {k}", f"{acc:.2%}")
                # Show a sample prediction
                sample_idx = 10 
                test_img = X_test[sample_idx].reshape(8, 8)
                pred = knn.predict([X_test[sample_idx]])
                
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(test_img, cmap='gray')
                ax.set_title(f"Pred: {pred[0]} (True: {y_test[sample_idx]})")
                ax.axis('off')
                st.pyplot(fig)

# --- Module 3: Linear Classifier Weights (CIFAR-style) ---
elif menu == "3. Linear Classifier Weights":
    st.header("🎨 Learned Templates (W Weights Visualization)")
    st.write("Each image below represents the 'template' learned by a linear classifier for a specific class.")
    
    classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer']
    cols = st.columns(len(classes))
    
    for i, name in enumerate(classes):
        # Generating synthetic 'template' weights
        # In a real CIFAR-10 model, these would be the reshaped rows of W
        np.random.seed(i)
        weight_template = np.random.randn(32, 32, 3)
        # Normalize to 0-1 for display
        weight_template = (weight_template - weight_template.min()) / (weight_template.max() - weight_template.min())
        
        with cols[i]:
            st.write(f"**{name}**")
            fig, ax = plt.subplots()
            ax.imshow(weight_template)
            ax.axis('off')
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("🧮 Loss Calculation Demo")
    loss_type = st.radio("Select Loss Function:", ["SVM (Hinge) Loss", "Cross-Entropy (Softmax) Loss"])
    
    # Dummy scores for demo
    scores = np.array([3.2, 5.1, -1.7])
    correct_class = 1 # 'Car'
    
    if loss_type == "SVM (Hinge) Loss":
        margins = np.maximum(0, scores - scores[correct_class] + 1.0)
        margins[correct_class] = 0
        loss = np.sum(margins)
        st.latex(r"L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)")
        st.write(f"Calculated SVM Loss: **{loss:.4f}**")
    else:
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        loss = -np.log(probs[correct_class])
        st.latex(r"L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)")
        st.write(f"Calculated Cross-Entropy Loss: **{loss:.4f}**")

# --- Module 4: Gradient Descent (SGD vs Momentum) ---
elif menu == "4. Gradient Descent (SGD vs Momentum)":
    st.header("📉 Optimization: SGD vs Momentum")
    
    lr = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
    mom = st.sidebar.slider("Momentum Coeff", 0.0, 0.99, 0.8)

    # Function: f(x) = x^2 * sin(x) for more interesting curves
    def f(x): return x**2 * np.sin(x)
    def df(x): return 2 * x * np.sin(x) + x**2 * np.cos(x)

    def run_opt(method="SGD"):
        x = -10.0
        path = [x]
        v = 0
        for _ in range(30):
            grad = df(x)
            if method == "Momentum":
                v = mom * v - lr * grad
                x += v
            else:
                x -= lr * grad
            path.append(x)
        return path

    path_sgd = run_opt("SGD")
    path_mom = run_opt("Momentum")

    fig, ax = plt.subplots(figsize=(10, 5))
    x_range = np.linspace(-12, 12, 100)
    ax.plot(x_range, f(x_range), 'k', alpha=0.3, label='Loss Landscape')
    ax.plot(path_sgd, [f(x) for x in path_sgd], 'ro-', label="Vanilla SGD", markersize=4)
    ax.plot(path_mom, [f(x) for x in path_mom], 'bo-', label="Momentum", markersize=4)
    ax.set_title("Optimization Path Comparison")
    ax.legend()
    st.pyplot(fig)
    
    st.info("Momentum usually helps dampening oscillations and speeds up convergence.")