import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def histogram_equalize(img_np, strength=1.0):
    hist, bins = np.histogram(img_np.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    eq_full = np.interp(img_np.flatten(), bins[:-1], cdf_normalized)
    eq_full = eq_full.reshape(img_np.shape)
    output = (1 - strength) * img_np + strength * eq_full
    return output.astype("uint8")

img = Image.open("testimage.jpg").convert("L")
img_np = np.array(img)
levels = [0, 0.33, 0.66, 1.0]
for s in levels:
    out = histogram_equalize(img_np, s)

    # If don't want to save file, use out
    filename = f"equalized_strength_{s:.2f}.png"
    Image.fromarray(out).save(filename)
    print(f"Saved: {filename}")

# Plots, can remove    
plt.figure(figsize=(12, 6))
for i, s in enumerate(levels):
    out = histogram_equalize(img_np, s)
    plt.subplot(2, 3, i + 1)
    plt.title(f"Strength {s}")
    plt.imshow(out, cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.show()