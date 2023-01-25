import streamlit as st 
from initial_conditions import provide_initial_condition
import make_sound as ms
import os



def sound_to_file():
    """ add a listener to every option and write a sound file created by the latest properties """
    c = st.session_state.c_slider
    waveform = st.session_state.wave_box
    p_o_b = st.session_state.pick_or_beat
    number_of_eigen_frequencies = st.session_state.noef
    MS.write_sound_to_file(c, number_of_eigen_frequencies, initial_func=waveform, pick_or_beat=p_o_b)
    

# Write a title
st.title('Sound of Surfaces')


surface = st.selectbox('Select Surface', ['Power Ellipsoid', 'Minor Ellipsoid','Major Ellipsoid'], key='surface_box', on_change=sound_to_file)
MS = ms.Make_Sound(surface)

number_of_eigen_frequencies = st.slider('Number of Overtones', 0, 100, step=1, value=20, key='noef')

waveform = st.selectbox('Select Waveform', ['cone', 'sawtooth', 'rectangle', 'cylinder'], key='wave_box', on_change=sound_to_file)

p_o_b = st.radio('Pick or Strike', ['pick', 'strike'], key='pick_or_beat')

c = st.slider('Propagation Velocity', 0.0, 0.6,step=0.01, value=0.1, key='c_slider', on_change=sound_to_file)
#print(c)

MS.write_sound_to_file(c, number_of_eigen_frequencies, initial_func='sawtooth', pick_or_beat='pick')

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

st.audio(audio_bytes, format='../audio/wav')


