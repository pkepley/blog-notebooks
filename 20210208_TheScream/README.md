# Recovering Waveform Images

This directory has a couple of scripts for recovering audio from
waveform images from [this
post](https://www.pakepley.com/2021-02-08-WineImagesIntoSound/). 

There are two scripts in this directory,

	1. `computerphile.py` - recovers audio from a waveform image as in [this Computerphile video](https://www.youtube.com/watch?v=VQOdmckqNro&feature=youtu.be).
	2. `wine.py` - kind of recovers audio from a waveform image from a wine bottle
	
In order to make this fully reproducible, you can use
[Pipenv](https://pipenv.pypa.io/en/latest/) to install the
requirements. The lock file has been included.

The scripts will open up a matplotlib window which you can click to
play the audio (:
