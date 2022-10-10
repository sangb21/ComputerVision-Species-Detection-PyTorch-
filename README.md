# [Rainforest Connecton Species Audio Detection Challenge](https://www.kaggle.com/competitions/rfcx-species-audio-detection/data)

- This was a machine learning competition to automate the detection species from their audio recordings. It was hosted on Kaggle. 
- A version of the notebook presented here was used to participate in the competition, and it ranked Top 6% ([64/1143](https://www.kaggle.com/sangayb))


# Aim

Given audio files that include sounds from numerous species, the objective is to use machine learning to predict species in the audio clip. 

# Data Description - Link to data is [here](https://www.kaggle.com/competitions/rfcx-species-audio-detection/data)
Each audio recording is 5 mins long. There are total of 3958 audio recording files availabe for training. 
Each single recorded audio contains signals from multiple species which are labelled by time-stamps `t_min` and `t_max`, where `t_min` indicates the start time the a particular species was heard and  `t_max` the time until the species was heard. Also provided are frequency range(`f_min`, `f_max`) of that particular siganl. 


Training data also includes false positive label occurrences to assist with training. 

Files: 

- `train_tp.csv` - training data of true positive species labels, with corresponding time localization.
- `train_fp.csv` - training data of false positives species labels, with corresponding time localization
- `train/` - the training audio files
- `test/` - the test audio files; the task is to predict the species found in each audio file

The meta file `train_tp.csv` contains the following columns:

- `recording_id` - unique identifier for recording
- `species_id` - unique identifier for species
- `songtype_id` - unique identifier for songtype
- `t_min` - start second of annotated signal
- `f_min` - lower frequency of annotated signal
- `t_max` - end second of annotated signal
- `f_max` - upper frequency of annotated signal


# Approach:

Past studies showed that rather than directly use the audio signals for species detection, instead using images of mel-spectrogram of the audio signals can be  used train deep learning models to perform species identification. The approach used in this notebook is the same. 


Since EDA indicates that each signal is approx. `2.64 sec` long. The plan is to chop each audio recordings (each `5 min` long) into `10 sec` long clips. 
Then plot the Mel-spectrogram, and save it as images which could be then be used to train deep learning models. There are a total of 24 different species that haven been labelled.  

This notebook illustrates how the audio recordings can be preprocessed to images of mel-spectrogram, and  how these images can used to detect species train a deep-learning model. 

