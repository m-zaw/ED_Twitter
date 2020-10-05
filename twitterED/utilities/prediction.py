import torch
import numpy as np
from tqdm.notebook import tqdm
# labels' codes
labels={'botdtct':{'bot': 0, 'human': 1},'emodtct':{'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3} }

def predict(dataloader_infr, model, device, predictions=None):
    
    if predictions is None:
        predictions = []

    for batch in tqdm(dataloader_infr):

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                 }
        #ignore/disable gradients
        with torch.no_grad():        
            outputs = model(**inputs)

        logits = outputs[0]     

        #detach from CPU means pulling values out of GPU to CPU
        #so we can use numpy
        logits = logits.detach().cpu().numpy()
                
        predictions.append(logits)
        
    predictions = np.concatenate(predictions, axis=0)
    preds_flat = np.argmax(predictions, axis=1).flatten()

        
    return preds_flat.tolist()