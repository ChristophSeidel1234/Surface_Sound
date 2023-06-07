import streamlit as st 
import numpy as np
from initial_conditions import provide_initial_condition
import make_sound as ms
import os
import matplotlib.pyplot as plt
import thinkdsp
import soundfile as sf
import scipy.signal as signal
import morphing
import math

st.set_option('deprecation.showPyplotGlobalUse', False)

def update_graphics():
    p = st.session_state.p_slider


def sound_to_file():
    """ add a listener to every option and write a sound file created by the latest properties """
    c = st.session_state.c_slider
    waveform = st.session_state.wave_box
    p_o_b = st.session_state.pick_or_beat
    number_of_eigen_frequencies = st.session_state.noef
    #MS.write_sound_to_file(c, number_of_eigen_frequencies, initial_func=waveform, pick_or_beat=p_o_b)
    

# Write a title
st.title('Sound of Surfaces')
st.sidebar.markdown("Choose options here:")

surface = st.sidebar.selectbox('Select Surface', ['Power Ellipsoid', 'Minor Ellipsoid','Major Ellipsoid'], key='surface_box', on_change=sound_to_file)
MS = ms.Make_Sound(surface)

number_of_eigen_frequencies = st.sidebar.slider('Number of Overtones', 0, 100, step=1, value=20, key='noef')

waveform = st.sidebar.selectbox('Select Waveform', ['cone', 'sawtooth', 'rectangle', 'cylinder'], key='wave_box', on_change=sound_to_file)

p_o_b = st.sidebar.radio('Pick or Strike', ['pick', 'strike'], key='pick_or_beat')

c = st.sidebar.slider('Propagation Velocity', 0.0, 0.6,step=0.01, value=0.1, key='c_slider', on_change=sound_to_file)
#print(c)

wave_surface = MS.write_sound_to_file(c, number_of_eigen_frequencies, initial_func='sawtooth', pick_or_beat='pick')


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

st.markdown("The pure surface sound")
st.audio(audio_bytes, format='../audio/wav')

hf = MS.get_highest_frequency(c, number_of_eigen_frequencies)
spectrum = wave_surface.make_spectrum(full=False)

fig, ax = plt.subplots()
x = spectrum.fs
y = spectrum.hs
#y = np.fft.fft(wave_surface.ys)
y = np.abs(y)
ax.set_title('Spectrum after Fourier-transform', fontstyle='italic')
ax.set_xlim([0, hf * 1.2])
ax.plot(x,y)
st.pyplot(fig)
#transform 
n = len(wave_surface.ys)
d = 1 / wave_surface.framerate
spectrum_domain = np.fft.fftfreq(n,d)
framerate = wave_surface.framerate
x_spec , y_spec = MS.get_spectrum( c, number_of_eigen_frequencies, initial_func='sawtooth', pick_or_beat='pick')



    
log_length = math.log(float(n))


p = st.sidebar.slider('p', 0.1, log_length, step=0.01, value=0.01, key='p_slider', on_change=update_graphics)
p = math.exp(p)


Spectrum = morphing.make_spectrum(x_spec, y_spec, spectrum_domain,framerate)

fig, ax = plt.subplots()
ax.set_title('Spectrum and Morphing Function', fontstyle='italic')
x = Spectrum.fs
y = Spectrum.hs
#gf = morphing.Global_Function(p, x_spec, y_spec, spectrum_domain)
#morphing.set_global_function(gf)
#global_func = gf.func
#length = len(x)
#convolution = morphing.smooth_func(global_func, p, length)
convolution = MS.create_morphing_func(c, number_of_eigen_frequencies, wave_surface, p, waveform, p_o_b)

ax.set_xlim([0, hf * 1.2])
ax.plot(x,y,x,convolution)

#st.write('x = ' + str(x))
#st.write('y = ' + str(y))
st.pyplot(fig)

# Get the directory where the app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the app directory
os.chdir(BASE_DIR)

#st.markdown("Upload an audio file here:")
#default
cwd = os.getcwd()
file_path = os.path.join(cwd, "data", "test.wav")

with open(file_path, "rb") as file:
    default_file = file.read()

# Load the input WAV file
data, sample_rate = sf.read(file_path)
#print(len(data))
#print(sample_rate)

uploaded_file = st.file_uploader(label="Choose a file", type=[".wav"])

if uploaded_file is None:
    uploaded_file = default_file

#st.sidebar.markdown("Or select a sample file here:")
st.markdown("uploaded sound")
st.audio(uploaded_file, format='../audio/wav')

#st.write(type(uploaded_file))
#st.write(uploaded_file)
recorded_wave = thinkdsp.read_wave_with_scipy('test.wav')
recorded_wave.unbias()
recorded_wave.normalize()

morphed_wave = MS.write_morphed_sound(c, number_of_eigen_frequencies, wave_surface,recorded_wave,p, waveform, p_o_b)

filename = os.path.join(desktop, "morphed_signal.wav")

if not os.path.exists(filename):
    os.mknod(filename)
audio_file = open(filename, 'rb')

audio_bytes = audio_file.read()

st.markdown("Morphed sound")
st.audio(audio_bytes, format='../audio/wav')

##wave = wave_surface * wave
#wave.plot() 
#plt.show()
spectrum = wave_surface.make_spectrum()



st.write('p = ' + str(p))
