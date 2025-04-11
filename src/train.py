from src.xor_dataset import x, y
from src.network import NeuralNetwork

nn = NeuralNetwork()


def run():
    epochs = 10000
    learning_rate = 0.1

    for epoch in range(epochs):
        output = nn.forward(x)
        nn.backward(x, y, output, learning_rate)

        if epoch % 1000 == 0:
            loss = ((y - output) ** 2).mean()  # Mean squared error
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # Final predictions
    print("\nFinal Predictions:")
    print(nn.forward(x))


if __name__ == "__main__":
    run()
