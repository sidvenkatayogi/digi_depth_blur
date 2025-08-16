import cv2
import torch
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def predict_depth(filename):
    s = datetime.now()
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img,(128,128))


    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    # plt.imshow(output)
    # for i in range(output.shape[0]):
    #     for j in range(output.shape[1]):
    #         plt.text(j, i, f"{output[i, j]:.0f}", ha="center", va="center", color="white", fontsize=4)
    e = datetime.now()
    print(f"Elapsed Time: {e - s}s")
    # plt.show()

    return output

def blur(filename, min_sigma=0.1, max_sigma=2.5):
    output = predict_depth(filename)

    # Load image and depth map (normalized 0..1)
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image,(128,128))
    depth = output

    # Map depth to sigma values (tune these)
    depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth_inverted = 1.0 - depth_normalized
    sigma_map = min_sigma + depth_inverted * (max_sigma - min_sigma)

    # Step 1: Precompute blurred images for discrete sigma values
    sigma_levels = np.linspace(min_sigma, max_sigma, 10)  # pick levels you want
    blurred_levels = [cv2.GaussianBlur(image, (0, 0), s) for s in sigma_levels]

    # Step 2: For each pixel, pick the two closest sigma levels and interpolate
    result = np.zeros_like(image, dtype=np.float32)

    sigma_levels_arr = np.array(sigma_levels, dtype=np.float32)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            sigma_val = sigma_map[y, x]

            # Find nearest two sigma levels
            idx = np.searchsorted(sigma_levels_arr, sigma_val)
            idx0 = max(0, idx - 1)
            idx1 = min(len(sigma_levels_arr) - 1, idx)

            if idx0 == idx1:
                result[y, x] = blurred_levels[idx0][y, x]
            else:
                s0, s1 = sigma_levels_arr[idx0], sigma_levels_arr[idx1]
                t = (sigma_val - s0) / (s1 - s0)
                result[y, x] = (1 - t) * blurred_levels[idx0][y, x] + t * blurred_levels[idx1][y, x]

    result = np.clip(result, 0, 255).astype(np.uint8)
    cv2.imwrite("variable_blur.jpg", cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result


if __name__ == "__main__":
    file = "shutterstock_1054656872-scaled.jpg"
    result = blur(file.strip())
    plt.imshow(result)
    plt.show()