# Emotion Detection on Twitter
<img src="https://github.com/ehsantaati/Twitter_PHD/blob/master/notebooks/figures/wordcld.png" alt="Word_cloud" width="50%" height="50%" >
Knowing what the public is thinking and feeling, is an important factor to respond properly to social events. Since COVID-19 pandemic outbreak, people have been expressing their thoughts and feeling about the pandemic on social networks. With analysing this data, authorities could target measures likely success of the health protection interventions that require public compliance to be effective.<br><br>
In this project, we benefitted from some Natural Language Processing (NLP) techniques, lexicon-based, supervised and unsupervised, to get insight from Twitter on COVID-19 pandemic. To this end, we analysed Wikipedia to extract the most recent related keywords to pull data from Twitter API and build the required dataset. Considering the main goal of the project, detecting and analysing emotions, we identified and filtered out tweets written by bots on Twitter. Afterwards, benefitting from a pre-trained version of Bidirectional Encoder Representations from Transformers BERT, we built a model to detect emotions in form of fear, anger, joy and sadness hidden in tweets. Finally, applying Latent Dirichlet Allocation (LDA) on each set of emotion, we concluded some motivations behind each set of emotion.<br><br>


**1.Extracting Covid-19 related keywords:**<br>
Talking it through with an expert, the following Wikipedia pages were used to extracted keywords:<br>
* [COVID-19 pandemic in the United Kingdom](https://en.wikipedia.org/wiki/COVID-19_pandemic_in_the_United_Kingdom)
* [NHS Test and Trace](https://en.wikipedia.org/wiki/NHS_Test_and_Trace)
* [Coronavirus disease 2019](https://en.wikipedia.org/wiki/Coronavirus_disease_2019)<br>

**2.Bot detection:**<br>
To detect tweets written by bots, we manually labelled a subset of pulled data. This dataset is accessible [here](https://github.com/ehsantaati/Twitter_PHD/blob/master/data/raw/tweet_df.csv)<br>

**3.Emotion Detection:**<br>
According to transfer learning rules, we retrained a pre-trained version of BERT ([“bert-large-uncased”](https://huggingface.co/bert-base-uncased)) in Pytorch on [WASSA-2017](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html) dataset.<br> 

**4.Topic Modelling:**<br>
Conducting LDA on each set of tweets, we extracted some topics with the highest contributions of each emotion set. These topics are being used for further interpretation to identify sets of triggers in each emotion set.
## [Environment](https://github.com/ehsantaati/Twitter_PHD/blob/master/requirements.yml):
* Pytorch
* Sckitlearn
## References:
* https://huggingface.co/bert-base-uncased
* http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
