# %%
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rnn_model import RNNModel
from time_series_dataset import TimeSeriesDataset
from input_data import InputData
from trainer import Trainer

# %%
data_path = "taller_volatilidad/data/trusted/currency_exchange.csv"
scaler = MinMaxScaler()
sequence_length = 8
train_size_proportion = 0.8

# %%
input_data = InputData(
    data_path=data_path,
    sequence_length=sequence_length,
    train_size_proportion=train_size_proportion,
    scaler=scaler,
)

# %%
input_data.df

# %%
X_train, X_test, y_train, y_test = input_data.create_training_and_test_sets()

# %%
# Hyperparameters
input_size = 1
hidden_size = 128
output_size = 1
learning_rate = 0.001
num_epochs = 200
batch_size = 10

# %%
# Create data loaders
train_dataset = TimeSeriesDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainer = Trainer(
    num_epochs=num_epochs,
    optimizer=optimizer,
    criterion=criterion,
    model=model,
    train_loader=train_loader,
    scaler=scaler,
)
trainer.train()

# %% [markdown]
# ## Evaluación sobre el conjunto de datos de prueba
# 

# %%
result = trainer.evaluate(X_test, y_test)
y_pred = result["y_pred"]
y_test = result["y_test"]

# %% [markdown]
# ## Gráficas de los valores predichos vs reales
# 

# %%
# Visualize predictions against actual data
df = input_data.df
train_size = input_data.train_size
plt.figure(figsize=(10, 6))
plt.plot(df.index[train_size + sequence_length :], y_test, label="Actual")
plt.plot(df.index[train_size + sequence_length :], y_pred, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Incoming Examinations")
plt.title("Incoming Examinations Prediction using RNN")
plt.legend()
plt.show()


