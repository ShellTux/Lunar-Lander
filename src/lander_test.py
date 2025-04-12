from lander import Lander
import os

from nn import NeuralNetwork

def test_serialization():
    """Test serialization and deserialization of Lander."""
    lander_original: Lander = Lander()
    lander_original.nn = NeuralNetwork(1, 2, 3)

    # Save the neural network to a file
    filename = 'test_lander.lander'
    lander_original.serialize(filename)

    lander_loaded = Lander.deserialize(filename)

    assert lander_original == lander_loaded

    # if os.path.exists(filename):
    #     os.remove(filename)
