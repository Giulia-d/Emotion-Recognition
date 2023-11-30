#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:50:01 2023

@author: giulia
"""

# Import packages
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import librosa
import torchaudio.transforms as T
from vit_pytorch import SimpleViT

import streamlit as st

from st_custom_components import st_audiorec
import gdown


# Function to load the pretrained model
@st.cache_resource
def load_model(path = None):
    # If a copy of the model is present on the pc, load it from the path
    # Download it otherwise
    if path is None:
        id = '1AFsJUsSICh7DwsRCQUGJiabvSQFIC_KQ'
        path = 'transformer_best.pth'
        gdown.download(id=id, output=path, quiet=False)
    
    model = SimpleViT(
        image_size = 448,
        patch_size = 64,
        num_classes = 8,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        channels = 1
    )
    
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


if __name__ == '__main__':
    
    # Set the tab bar
    gdown.download(id='11BI13Hyjz0Mfv0q6JiykLhxZkH_ZshPj', output='emotion.png', quiet=True)
    icon = Image.open("emotion.png")
    st.set_page_config(
        page_title="Emotion Recognition",
        page_icon=icon,
        layout="wide",
    )
       

    # Set title
    st.title(':red[Emotion Recognition]')
    
    # Important variables
    # Load the pretrained model, argument path of where the model is located, if none a copy will be download
    model = load_model()
    # Sample rate
    st.session_state['sr'] = 44100
    # millisecoonds
    st.session_state['ms'] = 5000
    # Number of samples in a record
    st.session_state['n_samples'] = int((st.session_state['sr']/1000) * st.session_state['ms'])
    # Windows length
    st.session_state['win_length'] = int((st.session_state['sr']/1000)*11)
    # List Classes
    st.session_state['classes'] = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
    # Spectogram
    st.session_state['spectrogram'] = T.MelSpectrogram(
                                          n_fft=1024,
                                          win_length=st.session_state['win_length'],
                                          hop_length=None,
                                          center=True,
                                          pad_mode="reflect",
                                          power=2.0,
                                      )
    # Transformation
    st.session_state['transform'] = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((64,448)),
                                        transforms.Normalize(
                                        mean=[0.485],
                                        std=[0.229]
                                        )])

    with st.container():
        
        # Radio button
        type_file = st.radio("What kind of audio file you want to recognize?",
            ('Upload Record', 'Live Record'))
        
        # Upload the file
        if type_file == 'Upload Record':
            
            st.header('Upload the audio file')
            
            #Create file uploader for the audio file
            file_audio = st.file_uploader("Choose an audio recording file", 
                                       accept_multiple_files = False,
                                       type = 'wav',
                                       help = 'Requires .wav file.')
            
            # Store it in session state
            if file_audio is not None:
                st.session_state['file_audio'] = file_audio
         
        # Record the audio from the app
        if type_file == 'Live Record':
            
            st.header('Record the audio')
                
            # Tool to register an audio from in the app
            # INFO: by calling the function an instance of the audio recorder is created
            # INFO: once a recording is completed, audio data will be saved as wav file
            audio_recording = st_audiorec()
    
            if audio_recording is not None:
                # Save the input on a file
                with open('myfile.wav', mode='bw') as f:
                    f.write(audio_recording)
                
                # Store it in session state
                st.session_state['file_audio'] = 'myfile.wav'
      
        
        
        # Read the wav file and prepare to give as input to the model
        if 'file_audio' in st.session_state:
            
            y, sr = librosa.load(st.session_state['file_audio'], mono = True, sr = st.session_state['sr'])
            
            # save the audio
            st.session_state['original_audio'] = y
            
            # Delete the wav file
            if os.path.exists('myfile.wav'):
                os.remove('myfile.wav')
            else:
                # Player to listen the audio
                st.audio(st.session_state['file_audio'])
                
            # Delete pre-existing audio
            if 'file_audio' in st.session_state:
                del st.session_state['file_audio']
            
            
            
            # Show the 1d signal
            fig_waveform, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(30,10))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.label_outer()
            ax.set_ylabel('Amplitude')
            st.pyplot(fig_waveform, clear_figure = True)
        
     
    # Button compute the prediction
    if st.button("Recognize emotion"):
        
        if 'original_audio' in st.session_state:
            
            y = st.session_state['original_audio']
            
            del st.session_state['original_audio']
            
            # replicate if audio is too short
            if len(y) < st.session_state['n_samples']:
                y = np.tile(y, st.session_state['n_samples'] // len(y) + 1)
    
            # Random crop to self.n_samples
            if len(y) > st.session_state['n_samples']:
              isrt = np.random.randint(0, len(y) - st.session_state['n_samples'])
              iend = isrt + st.session_state['n_samples']
              y = y[isrt:iend]
    
            # Perform transformation
            spec = st.session_state['spectrogram'](torch.Tensor(y))
            # convert to decibel units
            features = librosa.power_to_db(spec)[:, :896, np.newaxis]
            features = st.session_state['transform'](features)
            
            
            # Compute the prediction
            pred = np.argmax(model(features.unsqueeze(0)).detach().numpy(), axis=1)
            #st.write(pred)
            st.title('The predicted emotion is: :red[{}]'.format(st.session_state['classes'][int(pred)]))
          
        else: st.write('No file has been uploaded')
        
        