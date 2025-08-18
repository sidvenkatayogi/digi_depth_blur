import matplotlib.pyplot as plt
from main import blur
import numpy as np
from datetime import datetime

def compare_sigma_values():
    """
    Runs the blur function with different max_sigma values and displays the results.
    """
    image_file = "shutterstock_1054656872-scaled.jpg"  # Make sure this image exists
    focuses = np.linspace(0, 1.0, 5)

    results = []

    print("Generating blurred images for different focus values...")
    for focus in focuses:
        print(f"Running with focus = {focus}")
        s = datetime.now()
        result_image = blur(image_file, focus_level = focus)
        e = datetime.now()
        print(f"Elapsed time: {(e - s)}")
        results.append(result_image)
    print("Image generation complete.")

    # Create a plot to display the results
    num_results = len(results)
    fig, axes = plt.subplots(2, (num_results + 1) // 2, figsize=(15, 8))
    axes = axes.ravel() # Flatten the 2D array of axes

    for i, (res, focus) in enumerate(zip(results, focuses)):
        axes[i].imshow(res)
        axes[i].set_title(f"focus = {focus}")
        axes[i].axis('off')

    # Turn off any unused subplots
    for i in range(num_results, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("focus_comparison.jpg")
    plt.show()

if __name__ == "__main__":
    compare_sigma_values()
