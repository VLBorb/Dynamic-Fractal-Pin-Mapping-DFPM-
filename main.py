"""
FULLY INTEGRATED & EXPANDED HFBP SYSTEM (DFPM + FPIS + FHIDS)
Demonstrating 40 fractal pins, advanced metrics (MSE, PSNR, SSIM, Entropy),
holographic 3D storage with diagonal read, and pin evolution plots.

Author: (Adapted/Combined from previous references)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim_func
from scipy.fft import dctn, idctn

# ----------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# ----------------------------------------------------------------------

def dct2(block):
    """2D DCT with orthonormalization."""
    return dctn(block, type=2, norm='ortho')

def idct2(block):
    """2D iDCT with orthonormalization."""
    return idctn(block, type=2, norm='ortho')

def mse(imageA, imageB):
    """Mean Squared Error (MSE)."""
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    """Peak Signal-to-Noise Ratio (dB) based on MSE."""
    max_val = 1.0
    error = mse(imageA, imageB)
    if error == 0:
        return 100.0
    return 10.0 * np.log10((max_val ** 2) / error)

def calc_ssim(imageA, imageB):
    """
    Structural Similarity Index (SSIM).
    Uses skimage's structural_similarity.
    Both images should be float in [0..1].
    """
    return ssim_func(imageA, imageB, data_range=1.0)

def calc_entropy(image):
    """
    Compute a simple Shannon entropy (base 2) of an image's pixel distribution.
    This is a rough measure of complexity or 'fractal' randomness.
    """
    # Flatten the image, clip to [0,1], then bin
    vals = np.clip(image.ravel(), 0, 1)
    hist, bin_edges = np.histogram(vals, bins=256, range=(0,1), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def show_images(img_list, titles=None, cmap='gray'):
    """Display multiple images side by side."""
    count = len(img_list)
    plt.figure(figsize=(5*count, 4))
    for i, img in enumerate(img_list):
        plt.subplot(1, count, i+1)
        plt.imshow(np.clip(img, 0, 1), cmap=cmap)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()

def plot_pins_evolution(pins_before, pins_after):
    """
    Plots bar charts of pin values before and after tuning, side by side.
    pins_before and pins_after should be arrays/lists of length N with pin values.
    """
    n = len(pins_before)
    x_indices = np.arange(n)

    plt.figure(figsize=(12,5))
    plt.bar(x_indices - 0.2, pins_before, width=0.4, label='Before Tuning', alpha=0.7)
    plt.bar(x_indices + 0.2, pins_after, width=0.4, label='After Tuning', alpha=0.7)
    plt.xlabel('Pin Index')
    plt.ylabel('Pin Value [0..100]')
    plt.title('Fractal Pin Values Before & After FPIS Tuning')
    plt.xticks(x_indices, [str(i) for i in range(n)], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ----------------------------------------------------------------------
# 2. DATA STRUCTURES: FREQUENTIAL ADDRESS + FRACTAL PIN
# ----------------------------------------------------------------------

class FrequentialAddress:
    """
    Holds a dynamic frequential address in [0, 100],
    representing how 'strong' or 'active' a pin is.
    We also store a timestamp (could be fractal-based or integer).
    """
    def __init__(self, pin_id: int, initial_value: float):
        """
        :param pin_id: unique ID for the pin
        :param initial_value: a float in [0,100]
        """
        self.pin_id = pin_id
        self.value = float(np.clip(initial_value, 0.0, 100.0))
        self.timestamp = 0  # or fractal-based time index

    def update_value(self, new_val: float):
        self.value = float(np.clip(new_val, 0.0, 100.0))

    def get_value(self):
        return self.value

    def get_id(self):
        return self.pin_id

    def set_timestamp(self, ts: int):
        self.timestamp = ts

    def get_timestamp(self):
        return self.timestamp

    def __repr__(self):
        return f"FrequentialAddress(ID={self.pin_id}, value={self.value:.2f}, ts={self.timestamp})"

class FractalPin:
    """
    Higher-level pin that encapsulates:
      - FrequentialAddress
      - Additional fractal/frequential parameters if needed
    """
    def __init__(self, name: str, pin_id: int, initial_value: float):
        self.name = name
        self.freq_addr = FrequentialAddress(pin_id, initial_value)

    def set_value(self, val: float):
        self.freq_addr.update_value(val)

    def get_value(self):
        return self.freq_addr.get_value()

    def set_timestamp(self, ts:int):
        self.freq_addr.set_timestamp(ts)

    def get_timestamp(self):
        return self.freq_addr.get_timestamp()

    def get_id(self):
        return self.freq_addr.get_id()

    def __repr__(self):
        return (f"FractalPin(name={self.name}, id={self.get_id()}, "
                f"value={self.get_value():.2f}, ts={self.get_timestamp()})")

# ----------------------------------------------------------------------
# 3. DFPM (Dynamic Fractal Pin Mapping): BULK GENERATION
# ----------------------------------------------------------------------

class DFPM_BulkGenerator:
    """
    - Uses the aggregated pin values (0..100) to drive a fractal-like transform.
    - Here, we do a blockwise 2D-DCT. We threshold small coefficients based on
      an 'avg pin factor,' simulating fractal compression behavior.
    """
    def __init__(self, pins: list, block_size: int=8):
        self.pins = pins
        self.block_size = block_size

    def generate_bulk(self, base_image):
        """
        1) Convert the image to float32 [0..1].
        2) For each block, compute 2D-DCT, zero out coefficients below threshold.
        3) iDCT to reconstruct.
        4) The threshold is derived from the average pin value:
           threshold_strength = 0.1 * (1 - (avg_pin/100)).
        """
        base = base_image.astype(np.float32)
        h, w = base.shape
        out_img = np.zeros_like(base)

        # Compute average pin value
        avg_pin_val = np.mean([p.get_value() for p in self.pins])  # in [0..100]
        factor = avg_pin_val / 100.0
        threshold_strength = 0.1 * (1.0 - factor)

        # Process in 8x8 blocks
        for r in range(0, h, self.block_size):
            for c in range(0, w, self.block_size):
                block = base[r:r+self.block_size, c:c+self.block_size]
                # If partial block on edges, just copy as is or handle carefully
                if block.shape[0] < self.block_size or block.shape[1] < self.block_size:
                    out_img[r:r+self.block_size, c:c+self.block_size] = block
                    continue
                block_dct = dct2(block)
                block_dct[np.abs(block_dct) < threshold_strength] = 0.0
                block_idct = idct2(block_dct)
                out_img[r:r+self.block_size, c:c+self.block_size] = block_idct

        return out_img

# ----------------------------------------------------------------------
# 4. FPIS (Frequential Pin Identity System): FINE-TUNING
# ----------------------------------------------------------------------

class FPIS_AdaptiveTuner:
    """
    Fine-tunes each pin toward a chosen target_value (e.g. 85)
    by shifting each pin by 10% of the difference.
    This simulates an adaptive fractal-frequential 'local correction.'
    """
    def __init__(self, pins: list):
        self.pins = pins

    def tune_pins(self, target_value: float=85.0):
        for pin in self.pins:
            old_val = pin.get_value()
            delta = 0.1 * (target_value - old_val)
            pin.set_value(old_val + delta)

# ----------------------------------------------------------------------
# 5. FHIDS / Holographic 3D Storage SIMULATION
# ----------------------------------------------------------------------

class HolographicStorage3D:
    """
    Maintains a 3D array [depth, height, width].
    Each 'z' plane can store a 2D image.
    Diagonal read simulates a 'holographic retrieval' approach.
    """
    def __init__(self, depth:int, height:int, width:int):
        self.depth = depth
        self.height = height
        self.width = width
        self.storage = np.zeros((depth, height, width), dtype=np.float32)
        self.current_index = 0

    def store_bulk(self, image_2d: np.ndarray):
        """
        Store the given 2D array in the next available z-plane.
        If current_index exceeds depth, we overwrite (wrap-around).
        Returns the z-plane index used.
        """
        z = self.current_index % self.depth
        h, w = image_2d.shape
        if h > self.height or w > self.width:
            raise ValueError("Image too large for this holographic plane!")
        self.storage[z, :h, :w] = image_2d
        self.current_index += 1
        return z

    def diagonal_read(self, z_index: int):
        """
        Demonstration of 'diagonal read' from the z_index plane.
        We'll read from top-left to bottom-right diagonally.
        Returns a new 2D array with the same shape,
        filled in the order of diagonal traversal.
        """
        plane = self.storage[z_index]
        h, w = plane.shape
        output = np.zeros_like(plane)
        for diag_sum in range(h + w - 1):
            for r in range(max(0, diag_sum - (w-1)), min(h, diag_sum+1)):
                c = diag_sum - r
                output[r, c] = plane[r, c]
        return output

# ----------------------------------------------------------------------
# 6. HFBP PROCESSOR (INTEGRATING DFPM + FPIS + FHIDS)
# ----------------------------------------------------------------------

class HFBP_Processor:
    def __init__(self, pins: list, storage: HolographicStorage3D, block_size=8):
        """
        :param pins: List of FractalPin objects
        :param storage: HolographicStorage3D instance
        :param block_size: size for block-based transforms
        """
        self.pins = pins
        self.storage = storage
        self.dfpm_gen = DFPM_BulkGenerator(self.pins, block_size=block_size)
        self.fpis_tuner = FPIS_AdaptiveTuner(self.pins)

    def process_bulk(self, image_2d: np.ndarray, target=85.0):
        """
        Full flow:
          1) Generate fractal bulk from the original image (bulk_1).
          2) Store bulk_1 in HPC -> z1
          3) Fine-tune pins (FPIS).
          4) Re-generate fractal bulk from 'bulk_1' -> bulk_2
          5) Store bulk_2 -> z2
          6) Return z1, z2, and the final fractal bulk
        """
        # Bulk_1
        bulk_1 = self.dfpm_gen.generate_bulk(image_2d)
        z1 = self.storage.store_bulk(bulk_1)

        # Pin tuning
        self.fpis_tuner.tune_pins(target)

        # Bulk_2
        bulk_2 = self.dfpm_gen.generate_bulk(bulk_1)
        z2 = self.storage.store_bulk(bulk_2)

        return z1, z2, bulk_2

# ----------------------------------------------------------------------
# 7. MAIN DEMO - RUN EVERYTHING
# ----------------------------------------------------------------------

def main_demo():
    # 7.1 Create 40 PINS with random initial values
    np.random.seed(42)
    pins = []
    for i in range(40):
        name = f"Pin{i}"
        init_val = float(np.random.rand() * 100.0)
        p = FractalPin(name, i, init_val)
        p.set_timestamp(np.random.randint(0, 9999))  # random TS
        pins.append(p)

    # Store initial pin values for plotting
    pins_before = [pin.get_value() for pin in pins]

    print("INITIAL 40 PINS (SUMMARY):")
    for pin in pins:
        print(pin)
    print("=========================================")

    # 7.2 Load an image (256x256) for demonstration
    original_img = data.camera()
    original_img = img_as_float(original_img)  # [0..1]
    original_img = resize(original_img, (256, 256), anti_aliasing=True)
    original_img = original_img.astype(np.float32)

    # 7.3 Build a HolographicStorage3D
    depth, height, width = 10, 256, 256
    storage = HolographicStorage3D(depth, height, width)

    # 7.4 Create HFBP_Processor
    block_size = 8
    processor = HFBP_Processor(pins, storage, block_size=block_size)

    # 7.5 Run the process (target=85.0)
    z1, z2, final_bulk = processor.process_bulk(original_img, target=85.0)

    # Store final pin values for plotting
    pins_after = [pin.get_value() for pin in pins]

    # 7.6 Diagonal read from HPC
    diag_1 = storage.diagonal_read(z1)
    diag_2 = storage.diagonal_read(z2)

    # 7.7 Compute multiple metrics
    # MSE, PSNR, SSIM, Entropy
    final_mse = mse(original_img, final_bulk)
    final_psnr = psnr(original_img, final_bulk)
    final_ssim = calc_ssim(original_img, final_bulk)
    entropy_original = calc_entropy(original_img)
    entropy_final = calc_entropy(final_bulk)

    print(f"Final Bulk stored at z2={z2}")
    print(f"  MSE:   {final_mse:.4f}")
    print(f"  PSNR:  {final_psnr:.2f} dB")
    print(f"  SSIM:  {final_ssim:.3f}")
    print(f"  Entropy(original): {entropy_original:.3f}")
    print(f"  Entropy(final):    {entropy_final:.3f}")

    # 7.8 Visualization
    show_images(
        [original_img, final_bulk, diag_2],
        titles=[
            "Original",
            f"Final Bulk (z2)\nMSE={final_mse:.4f}, PSNR={final_psnr:.2f}, SSIM={final_ssim:.3f}",
            "Diagonal Read (z2)"
        ]
    )

    # 7.9 Plot pin evolution
    plot_pins_evolution(pins_before, pins_after)

    print("=========================================")
    print("FINAL 40 PINS (SUMMARY):")
    for pin in pins:
        print(pin)

    avg_pin_before = np.mean(pins_before)
    avg_pin_after  = np.mean(pins_after)
    print(f"\nAverage Pin Value: Before={avg_pin_before:.2f}, After={avg_pin_after:.2f}")

# If running in Colab or as a script, just call main_demo()
if __name__ == "__main__":
    main_demo()
