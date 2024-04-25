import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    GRU (Gated Recurrent Unit) model implemented in PyTorch.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        num_layers (int): The number of GRU layers.
        gru (nn.GRU): The GRU layer.
        fc (nn.Linear): The fully connected layer.

    Methods:
        forward(self, x):
            Performs the forward pass of the GRU model.

            Args:
                x (torch.Tensor): The input tensor with shape (batch_size, sequence_length, input_size).

            Returns:
                torch.Tensor: The output tensor with shape (batch_size, output_size).
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the GRU model.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor with shape (batch_size, output_size).
        """
        _, hidden = self.gru(x)
        output = self.fc(hidden[-1])
        return output

if __name__ == "__main__":
    # Test run
    batch_size = 5
    sequence_length = 12
    input_size = 15
    hidden_size = 10
    num_layers = 2
    output_size = 1

    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    input_data = torch.randn(batch_size, sequence_length, input_size)
    output = model(input_data)
    print(f"Output shape: {output.shape}")