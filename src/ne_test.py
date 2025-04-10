import numpy as np
from ne import NeuralNetwork

def test_network_shape():
    """Test the shape of the NeuralNetwork weights and biases."""
    nn = NeuralNetwork(2, 3, 3, 1)

    # Check the shape of the weights
    assert len(nn.weights) == 3, "Expected 3 weight matrices."
    assert nn.weights[0].shape == (2, 3), "First weight shape should be (2, 3)."
    assert nn.weights[1].shape == (3, 3), "Second weight shape should be (3, 3)."
    assert nn.weights[2].shape == (3, 1), "Third weight shape should be (3, 1)."

    # Check the shape of the biases
    assert len(nn.biases) == 3, "Expected 3 bias vectors."
    assert nn.biases[0].shape == (1, 3), "First bias shape should be (1, 3)."
    assert nn.biases[1].shape == (1, 3), "Second bias shape should be (1, 3)."
    assert nn.biases[2].shape == (1, 1), "Third bias shape should be (1, 1)."

def test_forward_shape():
    """Test the output shape of the forward method."""
    nn = NeuralNetwork(2, 3, 3, 1)

    # Manually set weights and biases
    weights = [
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),  # Shape (2, 3)
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),  # Shape (3, 3)
        np.array([[0.1], [0.2], [0.3]])  # Shape (3, 1)
    ]
    biases = [
        np.array([[0.1, 0.1, 0.1]]),  # Shape (1, 3)
        np.array([[0.1, 0.1, 0.1]]),  # Shape (1, 3)
        np.array([[0.1]])  # Shape (1, 1)
    ]

    nn.set_parameters(weights, biases)

    X = np.array([[0.1, 0.2]])  # Input shape (1, 2)

    output = nn.forward(X)

        # Calculate expected output manually
    layer1 = nn._sigmoid(X @ weights[0] + biases[0])  # Output from first layer
    layer2 = nn._sigmoid(layer1 @ weights[1] + biases[1])  # Output from second layer
    expected_output = nn._sigmoid(layer2 @ weights[2] + biases[2])  # Output from third layer

    # Forward pass through the network
    output = nn.forward(X)

    # Assertions
    assert output.shape == (1,), "Output shape should be (1,)."
    np.testing.assert_almost_equal(output, expected_output.flatten(), decimal=5,
                                   err_msg="Output should match expected output.")
