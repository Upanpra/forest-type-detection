from sklearn.metrics import confusion_matrix
import torch

# for calculating confusion matrix

def get_Ytrue_YPredict(net, data_loader):
    y_true = []
    y_predict = []
    net.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        if  torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)                           
        _, predicted = torch.max(outputs.data, 1)       
        
        predicted = predicted.view(-1).cpu().numpy()
        labels = labels.view(-1).cpu().numpy()
        # import pdb; pdb.set_trace()  # Python debugger
        y_true.extend([x for x in labels])
        y_predict.extend([x for x in predicted])
        total += 1
        #print(f"Processed {total}")
    net.train()
    
    return y_true, y_predict
