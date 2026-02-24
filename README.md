# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Stock price prediction is a challenging task due to the non-linear and volatile nature of financial markets. Traditional methods often fail to capture complex temporal dependencies. Deep learning, specifically **Recurrent Neural Networks (RNNs)**, can effectively model time-series dependencies, making them suitable for stock price forecasting.

* **Problem Statement**:
  Build an RNN model to predict the future stock price based on past stock price data.

* **Dataset**:
  A stock market dataset containing **historical daily closing prices** (e.g., Google, Apple, Tesla, or NSE/BSE data).
  The dataset is usually divided into **training and testing sets** after applying normalization and sequence generation.

  ## train data:
  <img width="674" height="945" alt="image" src="https://github.com/user-attachments/assets/83ab373c-bac8-4da7-8104-04687f2090f9" />

  ## test data:
  <img width="670" height="802" alt="image" src="https://github.com/user-attachments/assets/e1aa1855-fb37-46aa-818c-b235e17c52a1" />


## Design Steps

### Step 1:

Import required libraries such as `torch`, `torch.nn`, `torch.optim`, `numpy`, `pandas`, and `matplotlib`.

### Step 2:

Load the dataset (e.g., stock closing prices from CSV), preprocess it by **normalizing** values between 0 and 1, and create input sequences for training/testing.

### Step 3:

Define the **RNN model architecture** with an input layer, hidden layers, and an output layer to predict stock prices.

### Step 4:

Compile the model using **MSELoss** as the loss function and **Adam optimizer**.

### Step 5:

Train the model on the training data, recording training losses for each epoch.

### Step 6:

Test the trained model on unseen data and visualize results by plotting the **true stock prices vs. predicted stock prices**.

## Program
#### Name: MAHALINGA JEYANTH V
#### Register Number:212224220057

```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out

# Initialize Model, Loss, Optimizer
model = RNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the Model
epochs = 20
model.train()
train_losses = []
print('Name:MAHALINGA JEYANTH V')
print('Register number:212224220057')
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")

# Plot training loss
print('Name: MAHALINGA JEYANTH')
print('Register Number: 212224220057')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()


```

## Output

### True Stock Price, Predicted Stock Price vs time
<img width="1066" height="656" alt="image" src="https://github.com/user-attachments/assets/0ec4c585-b572-4dcd-8e74-aadd2911876f" />


### Predictions 

<img width="325" height="75" alt="image" src="https://github.com/user-attachments/assets/60540d92-94a9-44c0-8279-aee950ddd3e5" />


## Result
The **RNN model** was successfully implemented for **stock price prediction**.


