import streamlit as st 
from initial_conditions import provide_initial_condition
import make_sound as ms
import os
import pathlib



cwd = os.getcwd() 
print('Hallo000000000000000')
print(cwd)

#MS.write_sound_to_file(0.05, 60, initial_func='sawtooth', pick_or_beat='pick')


def sound_to_file():
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
print(number_of_eigen_frequencies)

waveform = st.selectbox('Select Waveform', ['sawtooth', 'cone', 'rectangle', 'cylinder'], key='wave_box', on_change=sound_to_file)
print(waveform)
p_o_b = st.radio('Pick or Beat', ['pick', 'beat'], key='pick_or_beat')
print(p_o_b)

#st.image('/Users/seidel/Desktop/repos/gaussian-ginger-student-code/surface_synth/synt_python/duck.png')
c = st.slider('Propagation Velocity', 0.0, 0.6,step=0.01, value=0.1, key='c_slider', on_change=sound_to_file)
print(c)

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


#import base64
#def add_bg_from_local(image_file):
#    with open(image_file, "rb") as image_file:
#        encoded_string = base64.b64encode(image_file.read())
#    st.markdown(
#    f"""
#    <style>
#    .stApp {{
#        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#        background-size: cover
#    }}
#    </style>
#    """,
#    unsafe_allow_html=True
#    )
#add_bg_from_local('/Users/seidel/Desktop/repos/gaussian-ginger-student-code/surface_synth/synt_python/duck.png')  

#st.markdown(
#   f"""
#   <style>
#   p {
#    background-image: url("/Users/seidel/Desktop/repos/gaussian-ginger-student-code/surface_synth/synt_python/duck.pngg");
#   }
#   </style>
#  """,
 #  unsafe_allow_html=True)

