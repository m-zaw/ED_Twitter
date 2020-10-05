from sklearn.metrics import f1_score
import numpy as np
from tqdm.notebook import tqdm
import torch

##THIS IS THE FUNCTION WE CHANGE. GOTTA HAVE SOFTMAX ARGMAX HERE
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1,).flatten() #why flatten? we dont want a list of lists, we just want a single array
    labels_flat = labels.flatten()
    #f1_score(labels_flat, preds, average='weighted')
    return  labels_flat, preds#weights classes according to its distribution. disgust with 6 classes is downweighted
#weighted vs macro 

def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        acc = len(y_preds[y_preds==label])/len(y_true) *100
        print(f'Accuracy: {acc:0.2f}')

#training evalution function
def evaluate(dataloader_val, model,device):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }
        #ignore/disable gradients
        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        #detach from CPU means pulling values out of GPU to CPU
        #so we can use numpy
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals