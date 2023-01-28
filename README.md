# Surface of Surfaces

This app is a software synthesizer that makes virtual surfaces sound.

## Run the App

* Clone the following [GitHub repository](https://github.com/ChristophSeidel1234/Surface_Sound)
* Open a terminal an go to your current working directory.
* Now we build an environment so that all used modules run conflict-free. Here is a way to do that with ANACONDA. For more information see the [CONDA CHEAT SHEET](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
    * create a new environment and select the python version \
    `conda create -n <environment name> python=3.10.9`
    * activate the environment\
    `conda activate <environment name>`
    * install all required modules\
    `pip install -r requirements.txt`

* run the app\
`streamlit run app.py`

## Instructions
We go step by step from top to bottom through the app.
* **Select Surface**\
I have chosen the shape of the surfaces (i.e. tuned them) so that the fundamental tone together with the first overtones form a major, minor or power cord.
<div style="display: flex;">
  <img src="images/major" width="33.33%" alt="image1">
  <img src="images/minor" width="33.33%" alt="image2">
  <img src="images/power" width="33.33%" alt="image3">
</div>


* **Number of Overtones**\
Here one can set the number of generalized harmonics. If you select more, the sound becomes more glassy.
* **Select Waveform**\
These are the different initial conditions. For example, `cone` means that one pulls out something like a tent at the surface, comparable with picking a guitar string.
* **Pick or Strike**\
Pick gives the location and strike the speed in the initial conditions. If you think of physical instruments, this would be the difference between a piano and a harpsichord.
* **Propagation Velocity**\
This means how fast is the speed of the wave on the surface. This is also like tuning an instrument, since the propagation velocity is coupled to the frequencies in the wave equation.

Finally, I would like to mention that each change of the above options writes or changes a file named `new_signal.wav` on your desktop with respect to the selected properties.

