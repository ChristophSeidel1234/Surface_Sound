import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
import morphing


domain = np.arange(999)
y_spectrum = np.zeros(999)
y_spectrum[123] = 34.
y_spectrum[150] = 50.
y_spectrum[400] = 32.
y_spectrum[660] = 30.



p = 1.
q = 1.


def update_graphics():
    p = st.session_state.p_slider
    q = st.session_state.q_slider
    

st.sidebar.markdown("Choose options here:")


p = st.sidebar.slider('p', 0.1, 999., step=1.0, value=1.0, key='p_slider', on_change=update_graphics)
q = st.sidebar.slider('q', 0.1, 999., step=1.0, value=1.0, key='q_slider', on_change=update_graphics)



x_spec = np.array([123.,150.,400.,660.])
y_spec = np.array([34.,50.,32.,30.])

gf = morphing.Global_Function(p, x_spec, y_spec, domain)
morphing.set_global_function(gf)
global_func = gf.func

fig, ax = plt.subplots()
ax.plot(domain, y_spectrum,domain,global_func)
st.pyplot(fig)

