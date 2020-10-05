import contractions
import re
import unicodedata
from transformers import BertTokenizer
from torch.utils.data import TensorDataset #setting up our dataset so it's usable in a pytorch environment
from sklearn.feature_extraction.text import CountVectorizer
import os


class text_cleaning:
    def __init__(self, text):
        self.text = text
    
    def fix_contractions(text):
        text = contractions.fix(text)

        return text
    
    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8','ignore')
        return text
    
    def remove_digits(text):
        text = re.sub(r'^\d+\s|\s\d+\s|\s\d+$',' ',text)
        
        return text

    # remove all punctuation
    def rm_punctuation1(text):
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text

    #remove punctuations except '?' and '!' using for emotion cleaning
    def rm_punctuation2(text):
        text = re.sub(r'[\'\"\.\\\/\,#]', '', text)
        text = re.sub(r'[^\w\s\?\!]', ' ', text)
        
        return text
        
    def remove_excess_whitespace(text):
        text = re.sub(r'\s{2,}', ' ', text).strip()
        
        return text

    def lower(text):
        text = text.lower()

        return text

    def rm_tags(text):
        text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text) 
        return text  
#set up a tokenizer object, using pre-trained BERT's own tokenizer
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)
## getting the length of text
def get_length(sent, max_len = 0,tokenizer=tokenizer):


    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    
    return len(input_ids)

## encode data with the defined tokeniser
def encoding_data(list_of_text, max_len,tokenizer=tokenizer):
    encoded_data = tokenizer.batch_encode_plus(
        list_of_text,
        add_special_tokens=True, #add the CLS and SEP tokens
        truncation=True,
        return_attention_mask=True, #gets the non-zero parts of the OHE vector of sentence?..so we know when to finish essentially
        pad_to_max_length=True,
        max_length=max_len,
        return_tensors='pt' #returns pytorch tensor
    )
    
    return encoded_data

#Most frequently occuring n-grams
def get_top_n_words(corpus, n=None, ngram_range=(1,1)):
    vec1 = CountVectorizer(ngram_range=ngram_range, 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

# getting the list of files inside a path
def get_files(root,ext = '.jpg',flist =None):
    if flist is None:
        flist = []
    for root,directories,filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(ext):
                fpath = os.path.join(root,filename)
                flist.append(fpath)
    return flist