import os
import pandas
import deep_beamline_simulation
from pathlib import Path
import matplotlib.pyplot as plt

# file dictionary for importing
im_data = (
    Path(deep_beamline_simulation.__path__[0]).parent
    / "deep_beamline_simulation/image_data"
)

images = []
for filename in os.listdir(im_data):
    im_data_path = im_data / filename
    images.append(im_data_path)

fig, axes = plt.subplots(1, 8)
fig.suptitle("Varying Radius of Surface Curvature", fontsize=20)

for i in range(0, len(images)):
    image = pandas.read_csv(images[i], skiprows=1)
    axes[i].imshow(image, aspect="auto")
plt.show()
