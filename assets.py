import torch
from torch import nn
from tqdm.auto import tqdm
import os

def train_mutilclass(model, dataset ,device, epochs,test_dataset=None ,Optimizer='SGD', learning_rate=0.01):
  
  try:
    from torchmetrics import Accuracy
  except Exception as e:
    if str(e)=="No module named 'torchmetrics'":
      os.system("pip install torchmetrics")
      from torchmetrics import Accuracy
    else:
      print(e)

  model=model.to(device)
  loss_fn=nn.CrossEntropyLoss()
  activation=nn.Softmax()
  accuracy_fn=Accuracy().to(device)

  if Optimizer=='adam':
    optimizer=torch.optim.Adam(params=model.parameters(), lr=learning_rate)
  else:
    optimizer=torch.optim.SGD(params=model.parameters(), lr=learning_rate)
  
  model.train()
  
  train_loss=0
  train_accuracy=0

  for epoch in tqdm(range(epochs)):
    for batch, (x, y) in enumerate(dataset):
      x_train=x.to(device)
      y_train=y.to(device)

      model.train()
      y_logits = model(x_train)
      y_preds=activation(y_logits).argmax(dim=1)
      loss=loss_fn(y_logits, y_train)
      accuracy=accuracy_fn(y_preds, y_train)

      train_loss += loss
      train_accuracy += accuracy

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    train_loss /= len(dataset)
    train_accuracy /= len(dataset)

  print(f'Epoch: {epoch} || Loss: {train_loss:.4f} || Accuracy: {train_accuracy*100:.4f} %')

