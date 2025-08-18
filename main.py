# TODO
# adjust focus depth (certain value in 0-1 is most clear), can use abs() to do this
# try defocusing
# fix depth edges bleeding into further depth blurred parts

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
    # plt.show()

    return output

def blur(filename, min_sigma=0.1, max_sigma=15.0, focus_level=0):
    output = predict_depth(filename)

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sq = int(np.sqrt(image.shape[0] * image.shape[1] / 250))
    # image = cv2.resize(image,(128,128))

    # Map depth to sigma values (tune these)
    # normalize (# 0 to 1.0 values, 0 is closest, 1.0 is farthest)
    depth = 1.0 - (output - np.min(output)) / (np.max(output) - np.min(output)) 
    sigma_map = min_sigma + abs(depth - focus_level) * (max_sigma - min_sigma)

    # Step 1: Precompute blurred images for discrete sigma values
    sigma_levels = np.linspace(min_sigma, max_sigma, 7)  # pick levels you want
    blurred_levels = [cv2.GaussianBlur(image, (0, 0), s) for s in sigma_levels]

    # Step 2: For each pixel, pick the two closest sigma levels and interpolate
    result = np.zeros_like(image, dtype=np.float32)

    sigma_levels_arr = np.array(sigma_levels, dtype=np.float32)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Define the window for averaging sigmas
            y_start = max(0, y - sq // 2)
            y_end = min(image.shape[0], y + sq // 2)
            x_start = max(0, x - sq // 2)
            x_end = min(image.shape[1], x + sq // 2)

            # Get the window of depths and sigmas
            depth_window = depth[y_start:y_end, x_start:x_end]
            sigma_window = sigma_map[y_start:y_end, x_start:x_end]

            # Get the current depth
            current_depth = depth[y, x]

            # Filter the window to include only pixels with higher depth
            higher_depth_mask = depth_window <= current_depth

            # Average the sigmas of these pixels
            if np.any(higher_depth_mask):
                sigma_val = np.mean(sigma_window[higher_depth_mask])
            else:
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