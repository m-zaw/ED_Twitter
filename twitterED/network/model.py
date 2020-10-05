from transformers import BertForSequenceClassification

def BERT(label_dict,weights ='bert-base-uncased'):
    loaded_model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels = len(label_dict),
        output_attentions = False, #dont need attendtion mask
        output_hidden_states = False #last layer before output)
    )
    return  loaded_model