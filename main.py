import cv2
import numpy as np
import pandas as pd
from scipy.fftpack import dct, idct

# DCT Encoder
class DCTEncoder:
    def __init__(self, block_size=8, threshold=10):
        self.block_size = block_size
        self.threshold = threshold
        self.quantization_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

    def encode(self, image):
        h, w = image.shape
        block_size = self.block_size
        bitstream = ""  # Binary string for the entire image
        
        # Process each block
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = image[i:i+block_size, j:j+block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')  # 2D DCT
                dct_block[np.abs(dct_block) < self.threshold] = 0  # Discard small coefficients
                quantized_block = np.round(dct_block / self.quantization_matrix)  # Quantize
                
                # Serialize the block as binary
                for value in quantized_block.flatten():
                    binary_value = format(int(value), "08b")
                    bitstream += binary_value
        
        # Group the binary string into 8-bit chunks
        grouped_bitstream = [bitstream[i:i+8] for i in range(0, len(bitstream), 8)]
        return grouped_bitstream, image.shape


# DCT Decoder
class DCTDecoder:
    def __init__(self, block_size=8):
        self.block_size = block_size
        self.quantization_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

    def decode(self, grouped_bitstream, shape):
        h, w = shape
        block_size = self.block_size
        
        # Flatten the grouped binary stream into a single string
        bitstream = "".join(grouped_bitstream)
        
        # Convert binary stream back to integer array
        quantized_values = [int(bitstream[i:i+8], 2) for i in range(0, len(bitstream), 8)]
        
        # Reshape into blocks
        num_blocks = (h // block_size) * (w // block_size)
        dct_blocks = np.array(quantized_values).reshape((num_blocks, block_size, block_size))
        
        # Reconstruct the image
        decoded_image = np.zeros((h, w), dtype=np.float32)
        block_idx = 0
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                quantized_block = dct_blocks[block_idx]
                dequantized_block = quantized_block * self.quantization_matrix  # Dequantize
                block = idct(idct(dequantized_block.T, norm='ortho').T, norm='ortho')  # Apply IDCT
                decoded_image[i:i+block_size, j:j+block_size] = block
                block_idx += 1
        
        return np.round(decoded_image).clip(0, 255).astype(np.uint8)


# Calculate Metrics
def calculate_metrics(original_image1, reconstructed_image1, grouped_bitstream1,
                      original_image2, reconstructed_image2, grouped_bitstream2):
    # Calculate MSE for both images
    mse1 = np.mean((original_image1 - reconstructed_image1) ** 2)
    mse2 = np.mean((original_image2 - reconstructed_image2) ** 2)
    SUM = mse1 + mse2
    
    # Calculate compression ratio (ρ)
    original_size_bits = 2 * 1200 * 1200 * 8
    compressed_size_bits = (len(grouped_bitstream1) + len(grouped_bitstream2)) 
    rho = compressed_size_bits / original_size_bits
    
    # Combined score
    score = SUM + (200 * rho)
    return SUM, rho, score


# Main Pipeline
def process_images(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Encode images
    encoder = DCTEncoder()
    grouped_bitstream1, shape1 = encoder.encode(image1)
    grouped_bitstream2, shape2 = encoder.encode(image2)
    
    # Decode images
    decoder = DCTDecoder()
    reconstructed_image1 = decoder.decode(grouped_bitstream1, shape1)
    reconstructed_image2 = decoder.decode(grouped_bitstream2, shape2)
    
    # Calculate metrics
    SUM, rho, score = calculate_metrics(image1, reconstructed_image1, grouped_bitstream1,
                                         image2, reconstructed_image2, grouped_bitstream2)
    print(f"MSE (SUM): {SUM}")
    print(f"Compression Ratio (ρ): {rho}")
    print(f"Combined Score: {score}")
    
    # Prepare submission
    size_image = 1200 * 1200
    size1 = max(0, size_image - len(grouped_bitstream1))
    size2 = max(0, size_image - len(grouped_bitstream2))
    submission = pd.DataFrame({
        "ID": list(range(size_image)),
        "CompressedImage1": grouped_bitstream1 + ["I"] * size1,
        "CompressedImage2": grouped_bitstream2 + ["I"] * size2,
        "Image1": reconstructed_image1.flatten(),
        "Image2": reconstructed_image2.flatten()
    })
    
    return submission, reconstructed_image1, reconstructed_image2


# Example Usage
image1_path = "Watermarked_Image1.tiff"
image2_path = "Watermarked_Image2.tiff"
submission, reconstructed_image1, reconstructed_image2 = process_images(image1_path, image2_path)

# Save submission file
submission.to_csv("submission.csv", index=False)
