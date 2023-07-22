import streamlit as st 
import numpy as np
import make_sound as ms
import os
import matplotlib.pyplot as plt
import thinkdsp
import soundfile as sf
import scipy.signal as signal
import scipy.io.wavfile as wav
from scipy.io import wavfile
import morphing
import math
import surface as surf

st.set_option('deprecation.showPyplotGlobalUse', False)
# Write a title
st.title('Sound of Surfaces')
# define sidebar
st.sidebar.markdown("Choose options here:")



#def update_morphing():
#    p = st.session_state.p_slider

def update_surfce():
    surface = st.session_state.surface_box
    noise = st.session_state.noise_box

def sound_to_file():
    """ add a listener to every option and write a sound file created by the latest properties """
    c = st.session_state.c_slider
    waveform = st.session_state.wave_box
    p_o_b = st.session_state.pick_or_beat
    number_of_eigen_frequencies = st.session_state.noef
    initial_value_domain = st.session_state.domain_slider
    p = st.session_state.p_slider

surface = st.sidebar.selectbox('Select Surface', ['Power Ellipsoid', 'Major Ellipsoid', 'Minor Ellipsoid'], key='surface_box', on_change=update_surfce)
number_of_eigen_frequencies = st.sidebar.slider('Number of Overtones', 0, 100, step=1, value=20, key='noef',on_change=sound_to_file)
waveform = st.sidebar.selectbox('Select Initial Shape', ['Cone', 'Cylinder'], key='wave_box', on_change=sound_to_file)
initial_value_domain = st.sidebar.slider('Initial Value Domain', 0.1, 1.,step=0.01, value=0.25, key='domain_slider', on_change=sound_to_file)
p_o_b = st.sidebar.radio('Pick or Hit', ['hit', 'pick'], key='pick_or_beat', on_change=sound_to_file)
c = st.sidebar.slider('Propagation Velocity / Tuning', 0.0, 1.0,step=0.01, value=0.1, key='c_slider', on_change=sound_to_file)




S = surf.Surface(surface)
initial_indices = S.initial_idxs
MS = ms.Make_Sound(S, number_of_eigen_frequencies,p_o_b)

wave_surface = MS.write_sound_to_file(c, initial_value_domain, waveform, p_o_b)
wave = wave_surface.copy()

wave_surface.normalize()

# Get the current user's home directory
home = os.path.expanduser("~")
# Set the path to the desktop
desktop = os.path.join(home, "Desktop")
# Set the full path to the file
filename = os.path.join(desktop, "new_signal.wav")

if not os.path.exists(filename):
    os.mknod(filename)
audio_file = open(filename, 'rb')
audio_bytes = audio_file.read()

st.markdown("The pure Surface Sound")
st.audio(audio_bytes, format='../audio/wav')

hf = MS.get_highest_frequency(c, number_of_eigen_frequencies)
spectrum = wave_surface.make_spectrum(full=True)

#transform 
n = len(wave_surface.ys)
d = 1 / wave_surface.framerate
spectrum_domain = np.fft.fftfreq(n,d)
framerate = wave_surface.framerate

log_length = math.log(float(n))
p = st.sidebar.slider('Morphing Width', 0.1, log_length, step=0.01, value=0.1*log_length, key='p_slider', on_change=sound_to_file)
p = math.exp(p)

noise = st.sidebar.selectbox('Select Noise', ['Pink Noise', 'White Noise', 'Brownian Noise', 'No Noise'], key='noise_box', on_change=update_surfce)

x_spec = MS.x_spec
y_spec = MS.y_spec
Spectrum = morphing.make_spectrum(x_spec, y_spec, spectrum_domain,framerate)

convolution = MS.create_morphing_func(c, wave_surface, p)
wave_envelope = MS.write_envelope_sound(c, wave_surface, p,noise)

# Get the current user's home directory
home = os.path.expanduser("~")
# Set the path to the desktop
desktop = os.path.join(home, "Desktop")
# Set the full path to the file
filename = os.path.join(desktop, "envelope_signal.wav")

if not os.path.exists(filename):
    os.mknod(filename)
audio_file = open(filename, 'rb')
audio_bytes = audio_file.read()

st.markdown("The Sound of the Surface surrounded by the Morphing Function which is equipped with a Noise")
st.audio(audio_bytes, format='../audio/wav')


# Get the directory where the app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the app directory
os.chdir(BASE_DIR)

#st.markdown("Upload an audio file here:")
#default

def update_file():
    uploaded_file = st.session_state.file

cwd = os.getcwd()
file_path = os.path.join(cwd, "recorded_sounds", "electric_violin.wav")

with open(file_path, "rb") as file:
    default_file = file.read()

# Load the input WAV file
data, sample_rate = sf.read(file_path)

uploaded_file = st.file_uploader(label="Choose a File",key='file', type=[".wav"])

def read_and_resample_wav_file(file_path):
    desired_sample_rate = 44100

    # Read the WAV file to get the audio data and current sample rate
    current_sample_rate, audio_data = wav.read(file_path)
    # Calculate the ratio of the desired sample rate to the current sample rate
    sample_rate_ratio = desired_sample_rate / current_sample_rate

    # Resample the audio data to the desired sample rate
    resampled_audio_data = signal.resample(audio_data, int(len(audio_data) * sample_rate_ratio))
    # take the left channel
    resampled_audio_data = resampled_audio_data[:, 0]
    time_array = np.arange(len(resampled_audio_data)) / desired_sample_rate
    wave = thinkdsp.Wave(resampled_audio_data,time_array,desired_sample_rate)
    return wave


recorded_wave = read_and_resample_wav_file(file_path)
#recorded_wave = thinkdsp.read_wave_with_scipy(file_path)
recorded_wave.unbias()


if uploaded_file is not None:
    # Create a directory to store the files if it doesn't exist
    os.makedirs("recorded_sounds", exist_ok=True)

    # Get the filename
    filename = uploaded_file.name

    # Specify the path to write the file
    file_path = os.path.join("recorded_sounds", filename)

    # Check if the file already exists
    if not os.path.exists(file_path):
        # Write the file to the specified path
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getvalue())
    recorded_wave = read_and_resample_wav_file(file_path)
    #recorded_wave = thinkdsp.read_wave_with_scipy(file_path)
    recorded_wave.unbias()

recorded_wave.normalize()



rec_spec = recorded_wave.make_spectrum()

st.markdown("Uploaded Sound")
st.audio(uploaded_file, format='../audio/wav')




morphed_wave = MS.write_morphed_sound(c, wave_surface,recorded_wave, p)

filename = os.path.join(desktop, "morphed_signal.wav")

if not os.path.exists(filename):
    os.mknod(filename)
audio_file = open(filename, 'rb')

audio_bytes = audio_file.read()

st.markdown("Morphed Sound")
st.audio(audio_bytes, format='../audio/wav')

x_rand, y_rand = MS.set_random_spectrum(c, number_of_eigen_frequencies)
Spec_rand = morphing.make_spectrum(x_rand, y_rand, spectrum_domain ,framerate)
random_wave = MS.write_random_sound_to_file(c, number_of_eigen_frequencies)
x_rand = Spec_rand.fs
y_rand = Spec_rand.hs


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))

## first figure
x1 = spectrum.fs
y1 = spectrum.hs
y1 = np.abs(y1)

axs[0].set_title('Surface Spectrum and Morphing Function', fontstyle='italic')
x2 = Spectrum.fs
y2 = Spectrum.hs
axs[0].set_xlim([0, hf * 1.2])
axs[0].plot(x2,y2,x2,convolution)


#second figure
#axs[0,1].set_title('Spectrum of the Recorded Sound', fontstyle='italic')
#x3 = rec_spec.fs
#y3 = np.abs(rec_spec.hs)

#axs[0,1].set_xlim([0, hf * 1.2])
#axs[0,1].plot(x3,y3)

#third figure
#axs[1].set_title('Recorded Spectrum', fontstyle='italic')
x3 = rec_spec.fs
y3 = np.abs(rec_spec.hs)
max_rec = np.max(y3)
max_spec = np.max(y2)
y2 = y2 / max_spec
y3 = y3 / max_rec

axs[1].set_title('Surface Spectrum and Recorded Spectrum', fontstyle='italic')
axs[1].set_xlim([0, hf * 1.2])
axs[1].plot(x2,y2,x3,y3)
axs[1].plot(x3,y3, color='red')


# fourth figure
#axs[1,1].set_title('Discrete White Noise', fontstyle='italic')
#axs[1,1].set_xlim([0, hf * 1.2])
#axs[1,1].plot(x_rand,y_rand)

st.pyplot(fig)


#filename = os.path.join(desktop, "random_signal.wav")

#if not os.path.exists(filename):
#    os.mknod(filename)
#audio_file = open(filename, 'rb')

#audio_bytes = audio_file.read()

#st.markdown("Random Sound")
#st.audio(audio_bytes, format='../audio/wav')
