Readme Emotion Recognition




The emotion recognition application allows us to understand the emotion of a person from a short recording of their voice.
The user can upload a pre-existing .wav file, from the upload button, or can record live their voice by clicking the radio button ‘Live record’ , the start recording button and at the end the stop button.
The waveform of the uploaded audio and then the user can proceed to analyze the signal through the neural model with the ‘Recognize emotion’ button. The prediction of the model is then displayed on the screen.
The predictable emotions are: neutral, calm, happy, sad, angry, fear, disgust and surprise.




The application is launched with the streamlit run command, if the computer is connected to the internet the application automatically downloads a copy of the model used for the prediction. If the user already has a copy the folder path needs to be defined in row 70 as the argument of the load_model() function.
The folder st_audiorec needs to be in the same path as the application.


The dataset use to train and test the model is https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio