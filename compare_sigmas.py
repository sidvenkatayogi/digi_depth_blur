import matplotlib.pyplot as plt
from main import blur
import numpy as np
from datetime import datetime

def compare_sigma_values():
    """
    Runs the blur function with different max_sigma values and displays the results.
    """
    image_file = "shutterstock_1054656872-scaled.jpg"  # Make sure this image exists
    sigma_values = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0]
    results = []

    print("Generating blurred images for different sigma values...")
    for sigma in sigma_values:
        print(f"Running with max_sigma = {sigma}")
        s = datetime.now()
        result_image = blur(image_file, max_sigma=sigma)
        e = datetime.now()
        # print({(e - s)strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]})
        print(f"Elapsed time: {(e - s)}")
        results.append(result_image)
    print("Image generation complete.")

    # Create a plot to display the results
    num_results = len(results)
    fig, axes = plt.subplots(2, (num_results + 1) // 2, figsize=(15, 8))
    axes = axes.ravel() # Flatten the 2D array of axes

    for i, (res, sigma) in enumerate(zip(results, sigma_values)):
        axes[i].imshow(res)
        axes[i].set_title(f"max_sigma = {sigma}")
        axes[i].axis('off')

    # Turn off any unused subplots
    for i in range(num_results, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("sigma_comparison.jpg")
    plt.show()

if __name__ == "__main__":
    compare_sigma_values()
