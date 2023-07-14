import streamlit as st 
import numpy as np
import make_sound as ms
import os
import matplotlib.pyplot as plt
import thinkdsp
import soundfile as sf
import scipy.signal as signal
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
initial_value_domain = st.sidebar.slider('Initial Value Domain', 0.1, 1.,step=0.01, value=0.1, key='domain_slider', on_change=sound_to_file)
p_o_b = st.sidebar.radio('Pick or Hit', ['pick', 'hit'], key='pick_or_beat', on_change=sound_to_file)
c = st.sidebar.slider('Propagation Velocity / Tuning', 0.0, 1.0,step=0.01, value=0.1, key='c_slider', on_change=sound_to_file)




S = surf.Surface(surface)
initial_indices = S.initial_idxs
MS = ms.Make_Sound(S, number_of_eigen_frequencies,p_o_b)
wave_surface = MS.write_sound_to_file(c, initial_value_domain, waveform, p_o_b)

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

x_spec = MS.x_spec
y_spec = MS.y_spec
Spectrum = morphing.make_spectrum(x_spec, y_spec, spectrum_domain,framerate)

convolution = MS.create_morphing_func(c, wave_surface, p)

# Get the directory where the app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the app directory
os.chdir(BASE_DIR)

#st.markdown("Upload an audio file here:")
#default

def update_file():
    uploaded_file = st.session_state.file

cwd = os.getcwd()
file_path = os.path.join(cwd, "data", "test.wav")

with open(file_path, "rb") as file:
    default_file = file.read()

# Load the input WAV file
data, sample_rate = sf.read(file_path)

uploaded_file = st.file_uploader(label="Choose a File",key='file', type=[".wav"])
recorded_wave = thinkdsp.read_wave_with_scipy('test.wav')
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
    recorded_wave = thinkdsp.read_wave_with_scipy(file_path)
    recorded_wave.unbias()

recorded_wave.normalize()

rec_spec = recorded_wave.make_spectrum()

st.markdown("Uploaded Sound")
st.audio(uploaded_file, format='../audio/wav')




morphed_wave = MS.write_morphed_sound(c, wave_surface,recorded_wave, p)
#st.write(f'len_rec = {len_rec}, len_conv = {len_conv}, len_morph = {len_morph}')

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


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))

# plot time signal:
# axs[0, 0].set_title("Signal")
# axs[0, 0].plot(t, s, color='C0')
# axs[0, 0].set_xlabel("Time")
# axs[0, 0].set_ylabel("Amplitude")

## first figure
x1 = spectrum.fs
y1 = spectrum.hs
y1 = np.abs(y1)
#axs[0, 0].set_title('Spectrum after Fourier-transform', fontstyle='italic')
#axs[0, 0].set_xlim([0, hf * 1.2])
#axs[0, 0].plot(x1,y1)

#second figure
axs[0,0].set_title('Surface Spectrum and Morphing Function', fontstyle='italic')
x2 = Spectrum.fs
y2 = Spectrum.hs
axs[0,0].set_xlim([0, hf * 1.2])
axs[0,0].plot(x2,y2,x2,convolution)

#second picture
axs[0,1].set_title('Spectrum of the Recorded Sound', fontstyle='italic')
x3 = rec_spec.fs
y3 = np.abs(rec_spec.hs)

axs[0,1].set_xlim([0, hf * 1.2])
axs[0,1].plot(x3,y3)

#third picture
axs[1,0].set_title('Recorded Spectrum', fontstyle='italic')
x3 = rec_spec.fs
y3 = np.abs(rec_spec.hs)
max_rec = np.max(y3)
max_spec = np.max(y2)
y2 = y2 / max_spec
y3 = y3 / max_rec

axs[1,0].set_title('Surface Spectrum and Recorded Spectrum', fontstyle='italic')
axs[1,0].set_xlim([0, hf * 1.2])
axs[1,0].plot(x2,y2,x3,y3)

axs[1,1].set_title('Discrete White Noise', fontstyle='italic')
axs[1,1].set_xlim([0, hf * 1.2])
axs[1,1].plot(x_rand,y_rand)

st.pyplot(fig)


filename = os.path.join(desktop, "random_signal.wav")

if not os.path.exists(filename):
    os.mknod(filename)
audio_file = open(filename, 'rb')

audio_bytes = audio_file.read()

st.markdown("Random Sound")
st.audio(audio_bytes, format='../audio/wav')
