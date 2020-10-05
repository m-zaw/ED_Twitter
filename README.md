# Emotion Detection on Twitter
<img src="https://github.com/ehsantaati/Twitter_PHD/blob/master/notebooks/figures/wordcld.png" alt="Word_cloud" width="50%" height="50%" >
Knowing what the public is thinking and feeling, is an important factor to respond properly to social events. Since COVID-19 pandemic outbreak, people have been expressing their thoughts and feeling about the pandemic on social networks. With analysing this data, authorities could target measures likely success of the health protection interventions that require public compliance to be effective.<br>
In this project, we benefitted from some Natural Language Processing (NLP) techniques, lexicon-based, supervised and unsupervised, to get insight from Twitter on COVID-19 pandemic. To this end, we analysed Wikipedia to extract the most recent related keywords to pull data from Twitter API and build the required dataset. Considering the main goal of the project, detecting and analysing emotions, we identified and filtered out tweets written by bots on Twitter. Afterwards, benefitting from a pre-trained version of Bidirectional Encoder Representations from Transformers BERT, we built a model to detect emotions in form of fear, anger, joy and sadness hidden in tweets. Finally, applying Latent Dirichlet Allocation (LDA) on each set of emotion, we concluded some motivations behind each set of emotion.

1.Extracting Covid-19 related keywords:
Talking it through with an expert, the following Wikipedia pages were used to extracted keywords:<br>
*COVID-19 pandemic in the United Kingdom(https://en.wikipedia.org/wiki/COVID-19_pandemic_in_the_United_Kingdom)

