import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Train perceptron for a logic gate
def train_gate(X, y, learning_rate=0.1, epochs=10000):
    weights = np.random.rand(X.shape[1])  # Initialize weights
    bias = np.random.rand(1)             # Initialize bias

    # Training loop
    for _ in range(epochs):
        for i in range(len(X)):
            input_data = X[i]
            target = y[i]
            # Forward pass
            output = sigmoid(np.dot(input_data, weights) + bias)
            # Update weights and bias
            error = target - output
            weights += learning_rate * error * input_data
            bias += learning_rate * error

    return weights, bias

# Test perceptron for a logic gate
def test_gate(X, weights, bias):
    results = []
    for i in range(len(X)):
        input_data = X[i]
        output = sigmoid(np.dot(input_data, weights) + bias)
        results.append((input_data, np.round(output)))
    return results

# Data for AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Data for OR gate
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# Train and test AND gate
weights_and, bias_and = train_gate(X_and, y_and)
print("Final weights for AND gate:", weights_and)
print("Final bias for AND gate:", bias_and)
print("Testing AND gate:")
for input_data, output in test_gate(X_and, weights_and, bias_and):
    print(f"Input: {input_data}, Output: {output}")

# Train and test OR gate
weights_or, bias_or = train_gate(X_or, y_or)
print("\nFinal weights for OR gate:", weights_or)
print("Final bias for OR gate:", bias_or)
print("Testing OR gate:")
for input_data, output in test_gate(X_or, weights_or, bias_or):
    print(f"Input: {input_data}, Output: {output}")
