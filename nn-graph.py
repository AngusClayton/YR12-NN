import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
a = np.array([0, 0, 1, 1])
b = np.array([0, 1, 0, 1])
y_xor = np.array([[0, 1, 1, 0]])
total_input = np.array([a, b])

# --- Network Architecture ---
input_neurons, hidden_neurons, output_neurons = 2, 2, 1
samples = total_input.shape[1]
lr = 0.1
np.random.seed(42)

# --- Weight Initialization ---
w1 = np.random.rand(hidden_neurons, input_neurons)
w2 = np.random.rand(output_neurons, hidden_neurons)

# --- Activation Function ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Forward Propagation ---
def forward_prop(w1, w2, x):
    z1 = np.dot(w1, x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# --- Backward Propagation ---
def back_prop(m, w1, w2, z1, a1, z2, a2, y):
    dz2 = a2 - y
    dw2 = np.dot(dz2, a1.T) / m
    dz1 = np.dot(w2.T, dz2) * a1 * (1 - a1)
    dw1 = np.dot(dz1, total_input.T) / m
    dw1 = np.reshape(dw1, w1.shape)
    dw2 = np.reshape(dw2, w2.shape)
    return dz2, dw2, dz1, dw1

# --- Training ---
losses = []
iterations = 10000
for i in range(iterations):
    z1, a1, z2, a2 = forward_prop(w1, w2, total_input)
    loss = -(1/samples) * np.sum(y_xor * np.log(a2) + (1 - y_xor) * np.log(1 - a2))
    losses.append(loss)
    da2, dw2, dz1, dw1 = back_prop(samples, w1, w2, z1, a1, z2, a2, y_xor)
    w2 = w2 - lr * dw2
    w1 = w1 - lr * dw1

# --- Plotting the Network Graph ---
def plot_network(w1, w2):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Neural Network Weights')

    # Node positions
    input_pos = [(0, i) for i in range(input_neurons)]
    hidden_pos = [(1, i - 0.5) for i in range(hidden_neurons)]
    output_pos = [(2, 0)]

    # Draw nodes
    ax.scatter(*zip(*input_pos), s=200, label='Input Layer')
    ax.scatter(*zip(*hidden_pos), s=200, label='Hidden Layer')
    ax.scatter(*zip(*output_pos), s=200, label='Output Layer')

    # Draw connections and weights (Input to Hidden)
    for i in range(input_neurons):
        for h in range(hidden_neurons):
            weight = w1[h, i]
            ax.plot([input_pos[i][0], hidden_pos[h][0]],
                    [input_pos[i][1], hidden_pos[h][1]],
                    'gray', lw=2 * abs(1))
            # Offset label to avoid overlap
            x_mid = (input_pos[i][0] + hidden_pos[h][0]) / 2
            y_mid = (input_pos[i][1] + hidden_pos[h][1]) / 2
            # Offset direction based on connection index
            offset = 0.18 * ((h - i) if input_neurons > 1 else 1)
            ax.text(x_mid + 0.08, y_mid + offset,
                    f'{weight:.2f}', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Draw connections and weights (Hidden to Output)
    for h in range(hidden_neurons):
        for o in range(output_neurons):
            weight = w2[o, h]
            ax.plot([hidden_pos[h][0], output_pos[o][0]],
                    [hidden_pos[h][1], output_pos[o][1]],
                    'gray', lw=2 * abs(1))
            # Offset label to avoid overlap
            x_mid = (hidden_pos[h][0] + output_pos[o][0]) / 2
            y_mid = (hidden_pos[h][1] + output_pos[o][1]) / 2
            offset = 0.18 * (h - (hidden_neurons-1)/2)
            ax.text(x_mid + 0.08, y_mid + offset,
                    f'{weight:.2f}', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Labels and layout
    ax.text(0, input_neurons - 0.5, 'Inputs', ha='center', va='center', fontsize=12)
    ax.text(1, hidden_neurons - 1, 'Hidden', ha='center', va='center', fontsize=12)
    ax.text(2, 0.5, 'Output', ha='center', va='center', fontsize=12)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Input', 'Hidden', 'Output'])
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.show()

plot_network(w1, w2)

# --- Prediction ---
def predict(w1, w2, input_data):
    z1, a1, z2, a2 = forward_prop(w1, w2, input_data)
    a2 = np.squeeze(a2)
    if a2 >= 0.5:
        print(f"For input {[i[0] for i in input_data]}, output is 1")
    else:
        print(f"For input {[i[0] for i in input_data]}, output is 0")

# --- Interactive Prediction ---
print("Weights for hidden layer:\n", w1)
print("Weights for output layer:\n", w2)

inp = input("> ")
while inp:
    try:
        test_input = inp.split(" ")
        a_val = int(test_input[0])
        b_val = int(test_input[1])
        test_data = np.array([[a_val], [b_val]])
        predict(w1, w2, test_data)
    except Exception as e:
        print("Error:", e)
    inp = input("> ")
