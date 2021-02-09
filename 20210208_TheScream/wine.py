from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.io.wavfile import write
import pygame


# make sure output directory is there
Path("./output").mkdir(exist_ok=True)

# get the input image, and fully threshold it
img_orig = cv2.imread("./input/thresh.jpg", 0)
(thresh, img_thresh) = cv2.threshold(img_orig, 128, 255, cv2.THRESH_BINARY)
ny, nx = img_thresh.shape

# extract envelope
envelope_low = np.zeros(nx)
envelope_high = np.zeros(nx)
for j in range(0, nx):
    tmp = np.where(img_thresh[:, j] == 0)[0]
    envelope_high[j] = tmp[0]
    envelope_low[j] = tmp[-1]


# form the signal by interleaving high and low envelope
signal = np.empty((2 * nx,), dtype=envelope_low.dtype)
signal[0::2] = envelope_low
signal[1::2] = envelope_high
ts = np.linspace(0, nx, 2 * nx)

# ma smoothing to remove ringing from interleaving
w = 2
signal2 = np.convolve(signal, np.ones(w), "valid") / w
signal2 = signal2 - signal2.mean()

# stretch signal to a more reasonable duration
stretch_factor = 8
ns = len(signal2)
signal3 = resample(signal2, int(stretch_factor * ns))
signal3 = signal3 / np.max(np.abs(signal3))
signal3 = np.iinfo(np.int16).max * signal3

# write to file
samplerate = 44100
write("./output/wine.wav", samplerate, signal3.astype(np.int16))


# mouse event for graph
def onclick(event):
    scream = pygame.mixer.Sound("./output/wine.wav")
    scream.play()
    pygame.time.wait(int(scream.get_length() * 1000))


# plot the result
fig, ax = plt.subplots(facecolor="k")
ax.imshow(img_thresh, cmap="gray_r")
ax.plot(envelope_low, label="Low Envelope", linewidth=0.5)
ax.plot(envelope_high, label="High Envelope", linewidth=0.5)
ax.set_xticks([])
ax.set_yticks([])
plt.legend(prop={"size" : 6}, loc = "lower right")

# output to file
ax.set_title("Converting the Wine Image", fontdict={"fontsize" : 10}, color="w")
plt.savefig("./output/wine_extraction.png", dpi=600, bbox_inches="tight")


# add event listener to play audio
pygame.mixer.init()
ax.set_title("Click Me to Hear the Audio")
fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()
