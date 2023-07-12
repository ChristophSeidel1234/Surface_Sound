# Sound of Surfaces

This app is a software synthesizer that makes virtual surfaces sound. To put it more precisely, we solve the wave equation on surfaces and additionally glue every conceivable recorded sound onto it.

## Run the App

* Clone the following [GitHub repository](https://github.com/ChristophSeidel1234/Surface_Sound)
* Open a terminal an go to your current working directory.
* Now we recommend to build a virtual environment so that all used modules run conflict-free. Here is a way to do that with `venv`. This is a Python buildt in environment that does not require any additional installation and has the advantage that it only exists locally in this folder. For more information see  [venv](https://docs.python.org/3/library/venv.html#module-venv)
    * Create a new environment\
    `python -m venv .venv`
    The last parameter, `.venv`, is the name of the directory to install the virtual environment into. You can name this whatever you would like.
    * Activate the environment\
    If you are on Windows, you will use `.venv\Scripts\activate.bat`\
    On other OSes, you will use source `.venv/bin/activate`
    * Install all required modules\
    `pip install -r requirements.txt`
    * Once you are finished, just use the `deactivate` command to exit the virtual environment.

* run the app\
`streamlit run app.py`

## Instructions
We go step by step from top to bottom through the app.
* **Select Surface**\
I have chosen the shape of the surfaces (i.e. tuned them) so that the fundamental tone together with the first overtones form a major, minor or power cord.
<div style="display: flex;">
  <img src="images/major.jpg" width="30%" alt="major">
  <img src="images/minor.jpg" width="30%" alt="minor">
  <img src="images/power.jpg" width="30%" alt="power">
</div>


* **Number of Overtones**\
Here one can set the number of generalized harmonics. If you select more, the sound becomes more glassy.
* **Select Initial Shape**\
These are the different shapes of the initial conditions. `Cone` means that one pulls out something like a tent at the surface, comparable with picking a guitar string, whereas `Cylinder` just means a cylindrical shape of the initial conditions.
* **Initial Value Domain**\
This slider indicates on which part of the surface the initial shape is defined. If it is close to zero, only one point is extracted, whereas if it is one, the initial shape is created on the entire surface.
* **Pick or Hit**\
Pick gives the location and Hit the speed in the initial conditions. If you think of physical instruments, this would be the difference between a piano and a harpsichord.
* **Propagation Velocity**\
This means how fast is the speed of the wave on the surface. This is also like tuning an instrument, since the propagation velocity is coupled to the frequencies in the wave equation.
* **Morphing Width**\
   Here you can specify how much the recorded sound should be morphed onto the surface. The construction is as follows: build a rectangular function      of width 2 * `Morphing Width` around the discrete spectrum of the surface, smooth this with a suitable mollifier via the convolution theorem,          multiply this with the spectrum of the recorded sound and send it back with the inverse Fourier transfom.\
   <img src="images/morphing_function.jpg" width="60%" alt="morphing_function">\
   This can also be seen as a very special filter that opens many small gates that nestle around the surface spectrum.

Finally, I would like to mention that each change of the above options writes or changes a file named `new_signal.wav` on your desktop with respect to the selected properties.

