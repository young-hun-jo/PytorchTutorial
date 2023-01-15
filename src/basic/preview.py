import torch
import torch.nn as nn

import numpy as np
import pandas as pd

# load dataset and preprocessing
dataset = pd.read_csv('./data/car_evaluation.csv')    # shape: (n_rows, 6)
cat_cols = dataset.columns[:-1]
for cat_col in cat_cols:
    dataset[cat_col] = dataset[cat_col].astype('category')

# concatenate categorical(numberized) nd-array
categorical_data = np.stack([
    dataset['price'].cat.codes.values,
    dataset['maint'].cat.codes.values,
    dataset['doors'].cat.codes.values,
    dataset['persons'].cat.codes.values,
    dataset['lug_capacity'].cat.codes.values,
    dataset['safety'].cat.codes.values
    ], axis=1)
outputs_data = dataset.output.astype('category').cat.codes.values    # label encoding format

# convert nd-array to torch.tensor
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)  # int64 <-> LongTensor
outputs_data = torch.tensor(outputs_data).type(torch.LongTensor)

# split train and test
total_records = dataset.shape[0]
test_records = int(total_records * 0.2)
categorical_train_data = categorical_data[:total_records - test_records]
categorical_test_data = categorical_data[total_records - test_records:]
outputs_train_data = outputs_data[:total_records - test_records]
outputs_test_data = outputs_data[total_records - test_records:]

# make embedding vector size from categorical data
categorical_column_sizes = [len(dataset[column].cat.categories) for column in cat_cols]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2))for col_size in categorical_column_sizes]


# make Custom model
class Model(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, p=0.4):
        super(Model, self).__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ftr_n, embed_n) for ftr_n, embed_n in embedding_size])
        self.embedding_dropout = nn.Dropout(p)

        all_layers = []
        total_embed_n = sum(embed_n for _, embed_n in embedding_size)
        input_size = total_embed_n

        for i in hidden_size:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))  # input(or hidden ouput) inplace시킴을 의미
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(hidden_size[-1], output_size))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        embeddings = []
        for i, embed_layer in enumerate(self.all_embeddings):
            embeddings.append(embed_layer(x[:, i]))  # x[:,i] shape: (batch_size, 2)
        x = torch.cat(embeddings, 1)   # concatenate embedding shape: (batch_size, 12) 2 * feautre개수(6개)
        x = self.embedding_dropout(x)
        x = self.layers(x)
        return x


# define model
model = Model(categorical_embedding_sizes, output_size=4, hidden_size=[200, 100, 50], p=0.4)

# train
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500
loss = []
for i in range(epochs):
    y_pred = model(categorical_train_data)
    single_loss = loss_fn(y_pred, outputs_train_data)
    loss.append(single_loss)

    if (i+1) % 25 == 0:
        print(f'Epoch: {i+1} -> Loss: {single_loss.item(): .3f}')

    # clear gradients
    optimizer.zero_grad()
    # back-propagation based on loss and calculate gradients
    single_loss.backward()
    # update parameters based on gradients
    optimizer.step()

# Test
with torch.no_grad():
    y_val = model(categorical_test_data)
    val_loss = loss_fn(y_val, outputs_test_data)
print(f"Validation Loss for test: {val_loss: .3f}")

