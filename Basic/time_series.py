import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

# start = time.time()
flights = sns.load_dataset("flights")

#%%
plt.figure(figsize=(10,4))
plt.xlabel('Months')
plt.ylabel('Number of Passengers')
plt.title('Months vs Passengers')
plt.grid(True)
plt.plot(flights['passengers'])
plt.show()
#%%

all_data = flights['passengers'].astype('float').values

test_data_size = 12
train = all_data[:-test_data_size]
test = all_data[-test_data_size:]

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train.reshape(-1,1))
train_norm = scaler.transform(train.reshape(-1, 1))

#%%

train_norm = torch.FloatTensor(train_norm).view(-1)

#%%
window_size = 12

output_size = 1
def create_windows_labels(data, window, output_size):
    input_seq = []
    L = len(data)
    for i in range(L-window-output_size+1):
        seq = data[i:i+window]
        label = data[i+window:i+window+output_size]
        input_seq.append((seq, label))
    return input_seq

train_seq = create_windows_labels(train_norm, window_size, output_size)
#%%

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=200, output_size=output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1 ,1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

#%%
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)
#%%

epochs = 300

for i in range(epochs):
    for seq, labels in train_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)

        loss = loss_function(y_pred, labels)
        loss.backward()
        optimizer.step()

    if i%25 == 0:
        print(f'epoch: {i:3} loss: {loss.item():10.8f}')

# print(f'epoch: {i:3} loss: {loss.item():10.10f}')

#%%
fut_pred = 24
test_inputs = train_norm[-12:].tolist()

#%%

for i in range(fut_pred//output_size):
    seq = torch.FloatTensor(test_inputs[-window_size:])
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs += model(seq).data.numpy().tolist()

#%%

actual_predictions = scaler.inverse_transform(np.array(test_inputs[window_size:] ).reshape(-1, 1))
print(actual_predictions)

#%%

plt.figure(figsize=(10,4))
plt.xlabel('Months')
plt.ylabel('Number of Passengers')
plt.title('Months vs Passengers')
plt.grid(True)
plt.plot(flights['passengers'])
plt.plot(np.arange(132, 156), actual_predictions)
plt.show()

# end = time.time()

# print("Time to run the code: " + str(end-start) + " sec")