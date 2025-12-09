import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

img = Image.open("testimage.jpg").convert("L")
img_np = np.array(img)

kernel_sizes = [13, 53, 73, 93]
os.makedirs("median_outputs_cv2", exist_ok=True)

filtered_images = []

for k in kernel_sizes:
    print(f"Applying median filter with kernel {k}")
    filtered = cv2.medianBlur(img_np, k)
    filtered_images.append((k, filtered))

    # saving files, can use filtered_images 
    # to take out arrays directly
    filename = f"median_outputs_cv2/median_k{k}.png"
    cv2.imwrite(filename, filtered)
    print(f"Saved: {filename}")

#Plots, can remove
plt.figure(figsize=(4 * (len(kernel_sizes) + 1), 4))
plt.subplot(1, len(kernel_sizes) + 1, 1)
plt.imshow(img_np, cmap="gray")
plt.title("Original")
plt.axis("off")
for i, (k, filtered) in enumerate(filtered_images):
    plt.subplot(1, len(kernel_sizes) + 1, i + 2)
    plt.imshow(filtered, cmap="gray")
    plt.title(f"Kernel {k}")
    plt.axis("off")
plt.tight_layout()
plt.show()