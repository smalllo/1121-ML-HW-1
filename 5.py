import numpy as np
import matplotlib.pyplot as plt

def target_function(x1, x2):
    return 2 * x1 - 3 * x2 + 1


np.random.seed(0)
num_samples = 40
X = np.random.rand(num_samples, 2) * 10 - 5 
Y = np.sign(target_function(X[:, 0], X[:, 1]))  

w = np.zeros(3)  


def plot_data_and_boundary(X, Y, w):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], label="Class 1", marker='o')
    plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], label="Class -1", marker='x')
    

    x1_boundary = np.linspace(-5, 5, 100)
    x2_boundary = (-w[0] - w[1] * x1_boundary) / w[2]
    
    plt.plot(x1_boundary, x2_boundary, 'k-', label="Decision Boundary")
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Perceptron Learning Algorithm")
    plt.legend()
    plt.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()

# Plot initial data points
plot_data_and_boundary(X, Y, w)

# Perceptron learning algorithm
learning_rate = 0.1
num_epochs = 100

for epoch in range(num_epochs):
    misclassified = 0
    for i in range(num_samples):
        x = np.concatenate(([1], X[i]))  
        if Y[i] * np.dot(w, x) <= 0:
            w += learning_rate * Y[i] * x
            misclassified += 1
    if misclassified == 0:
        break


plot_data_and_boundary(X, Y, w)
