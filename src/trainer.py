from typing import Optional
from torch import nn
import torch


SUPPORTED_LR_SCHEDULERS = {"cosine"}
def l1_reg(model: torch.nn.Module) -> torch.Tensor:
    # L1 regularization term
    l1_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l1_reg = l1_reg + torch.norm(param, p=1)
    return l1_reg


def trainCNN(net, train_loader, test_loader,
          num_epochs, learning_rate,
          compute_accs=True, lr_scheduler: Optional[str] = "cosine", weight_decay: Optional[str] = "l1"):
    if weight_decay is not None:
        assert weight_decay in ["l1", "l2"], weight_decay  # Note can add l2 later
        wd_lambda = 1e-2
    else:
        wd_lambda = None
  
    # First initialize the criterion (loss function) and the optimizer
    # (algorithm like gradient descent). Here we use a common loss function for multi-class
    # classification called the Cross Entropy Loss and the popular Adam algorithm.
    criterion = nn.CrossEntropyLoss()
    if weight_decay is None or weight_decay == "l1":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif weight_decay == "l2":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay = wd_lambda)
    else:
        raise ValueError(f"`weight_decay should be l1, l2, or None, but got {weight_decay}")
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay = 1e-2)
    
    if lr_scheduler is not None:
        assert lr_scheduler in SUPPORTED_LR_SCHEDULERS, f"{lr_scheduler} unsupported. Not in {SUPPORTED_LR_SCHEDULERS}"
        if lr_scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        else:
            raise NotImplementedError(f"lr scheduler {lr_scheduler} not in supported set {SUPPORTED_LR_SCHEDULERS}")

    train_accs = []
    test_accs = []
    
    # below is the code I am adding to test the train_loader values
    #print(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader): # Loop over each batch in train_loader    
            # If you are using a GPU, speed up computation by moving values to the GPU
            if torch.cuda.is_available():
                net = net.cuda()
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()               # Reset gradient for next computation
            outputs = net(images)               # Forward pass: compute the output class given a image
            loss = criterion(outputs, labels)   # Compute loss: difference between the pred and true

            if weight_decay == "l1":
                # Add the L1 regularization term to the loss
                loss = loss + wd_lambda * l1_reg(model=net)

            loss.backward()                     # Backward pass: compute the weight
            optimizer.step()                    # Optimizer: update the weights of hidden nodes

            if (i + 1) % 100 == 0:  # Print every 100 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        if compute_accs: 
          train_acc = accuracy(net, data_loader= train_loader)
          test_acc = accuracy(net, data_loader= test_loader)
          train_accs.append(train_acc)
          test_accs.append(test_acc)
          print(f'Epoch [{epoch + 1}/{num_epochs}], Train Acc {100 * train_acc:.2f}%, Test Acc {100 * test_acc:.2f}%')
        if lr_scheduler is not None:
            lr_scheduler.step()
    if compute_accs:
        return net, train_accs, test_accs
    else:
        return net
   


def accuracy(net, data_loader):
    net.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        if  torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)                           # Make predictions
        _, predicted = torch.max(outputs.data, 1)       # Choose class with highest scores
        total += labels.size(0)                         # Increment the total count
        correct += (predicted == labels).sum().item()   # Increment the correct count
    net.train()
    return correct / total


def plot_history(histories):
    plt.figure(figsize=(16,10))
    epochs = list(range(1, len(histories[0]['train_accs']) + 1))
    for model_history in histories:
      val = plt.plot(epochs, model_history['test_accs'],
                     '--', label=model_history['name'] + ' Test')
      plt.plot(epochs, model_history['train_accs'], color=val[0].get_color(),
               label=model_history['name'] + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim([1,max(epochs)])
