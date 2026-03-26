# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset


An LSTM-based model for recognizing named entities is a type of neural network that uses Long Short-Term Memory (LSTM) layers to identify and classify proper names and entities within a text, such as person names, locations, organizations, dates, etc. It is commonly employed in Named Entity Recognition (NER) tasks because LSTMs are effective at capturing sequential dependencies and context within text. Typically, these models process tokenized input data, learn contextual representations, and output labels for each token indicating whether it belongs to a specific entity type. This approach improves the accuracy of extracting meaningful information from unstructured text data

<img width="636" height="715" alt="image" src="https://github.com/user-attachments/assets/2066a058-64b5-41b7-b1fd-8e0e767ae5f7" />


## DESIGN STEPS
### STEP 1: 
Load data, create word/tag mappings, and group sentences.

### STEP 2: 

Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.

### STEP 3: 

Define dataset and DataLoader for batching.

### STEP 4: 

Build a bidirectional LSTM model for sequence tagging.

### STEP 5: 

Train the model over multiple epochs, tracking loss.


### STEP 6: 



Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.

## PROGRAM

### Name:R.SUBHASHRI

### Register Number:212223230219

```python
# Model definition
class BiLSTMTagger(nn.Module):

    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.fc(x)

model = BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
                val_loss += loss.item()
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")

    return train_losses, val_losses


```

### OUTPUT

## Loss Vs Epoch Plot

<img width="670" height="523" alt="image" src="https://github.com/user-attachments/assets/3942fcbb-1a11-4aee-9dac-79a45a337140" />


### Sample Text Prediction

<img width="370" height="388" alt="image" src="https://github.com/user-attachments/assets/a56c8a38-d7f6-4dbd-a5fe-30a0a3d533eb" />


## RESULT

Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
