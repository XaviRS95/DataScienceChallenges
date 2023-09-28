# DataScienceChallenges

This is a collection of all the challenges that I will take in order to learn and grasp all the concepts in the areas of Machine and Deep Learning. 

## Classification Challenges:

### Horses Health:
https://github.com/XaviRS95/DataScienceChallenges/tree/main/Classification/Horses-Health

Kaggle competition where you have to predict the outcome of a horse (lived, died or euthanized) using the provided dataset using f1-score. The URL for this challenge is: https://www.kaggle.com/competitions/playground-series-s3e22 

So far, my highest punctuation has been of 0.78 using RandomForest. There are still some feature engineering and other techniques that I want to try, since other people have been able to obtain 0.85.

When using classification_report, I have noticed that 1 category has a lower precision than the rest (around 0.1), while another has the same problem with the recall. I'm currently working on what is causing this imbalance and how I can solve it, since it might be the key to improve the f1-score to 0.8