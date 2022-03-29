# Using Etiquette Strategies to Mitigate Negative Emotions in Computer-Based Job Interviews

Welcome to my final year dissertation project. 

###### tl;dr Can we use human-automation etiquette strategies (i.e. different comment styles) to reduce negative emotions in computerised interviews? Let's evaluate facial expressions and speech and see.

Different machine learning techniques have been investigated to identify affect in users in order that technology can adapt to optimise user experience. There has been limited research conducted regarding adapting a user interface in a computer interview context, particularly that reducing negative emotions which could impede candidate performance.

The candidate uses a user interface as shown below, speaks through their thought process, clicks a final answer and 'Submit'.

<p align="center">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/44368206/160666801-8c8a7e24-d423-4c6e-ab08-c7c5d34848ce.png">
</p>


As the user speaks through their answers, facial expressions and speech are analysed. Once frustration is detected, a comment is shown on the bottom right - either a

**_bald_** comment - direct, taking no account into the level of imposition to the hearer

or a

**_positive politeness_** comment - minimises imposition between the speaker and hearer with statements of solidarity and compliments.

<p align="center">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/44368206/160670969-fa9024ce-c42a-4285-80f4-4e5c3ec66127.png">
</p>

### Facial Expression Recognition
Key tools used in facial expression recognition include Keras (Chollet et al. 2015), the FER-2013 dataset (Goodfellow et al. 2013) and the open source computer vision library OpenCV (Bradski 2000). Google Colaboratory was used for model training because of its access to CUDA, providing GPUs for processing.
The valence-arousal scale is a common way to taxonomise emotions (Buechel & Hahn 2016).

See the CNN architecture used below:
<p align="center">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/44368206/160673398-7b68c6a9-77d0-4802-b80b-2dcc61bf09c6.png">
</p>

See the final [project paper](projectpaper.pdf) for further details, including on the architecture.

### Speech analysis
For speech emotion recognition we utilised the RAVDESS dataset (Livingstone & Russo 2018) which contains 7356 audio (speech and song) and visual recordings of 12 male and 12 female actors pronouncing English sentences with expressions in the categories calm, happy, sad, angry, fearful, surprise, and disgust. The dataset has been used extensively in research, for applications such as the evaluation of machine learning models such as multitask learning (Zhang et al. 2016).

The speech architecture uses one dimensional convolutional layers, since two dimensions are more suitable for images. 

See the CNN architecture used below:
<p align="center">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/44368206/160675270-b7f5c951-d714-43ab-9c91-190369e02816.png">
</p>

See the final [project paper](projectpaper.pdf) for further details, including on the architecture.

### Conclusions
I successfully discovered a trend indicating that frustration in a computer automated interview negatively impacts task performance. This extends previous similar findings in the context of human controlled robots and tutoring systems (Yang & Dorneich 2015) (Yang & Dorneich 2018). Additionally, the work provides evidence that both bald and positive politeness comments reduce negative affect in participants. In the majority of cases, positive politeness comments proved most effective.
