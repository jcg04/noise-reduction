# Noise-reduction
Remove noise from images while preserving details.

## Content
1. What are noise in images?
2. Task Dfinitions.
3. Solution Key Ideas.
4. Implementation.
5. Result Examples.
6. Conclusion.
7. Further Improvements.

## 1. What are noise in images?
Noise in images refers to unwanted random variations in pixel intensity values, which obscure or distort the visual details of an image. It is often introduced during image acquisition, transmission, or processing. Noise can make it difficult to extract meaningful information from an image, particularly in applications like computer vision, medical imaging, or photography.
### Types of Noise in Images
#### Gaussian Noise:
- Description: Caused by random variations following a normal (Gaussian) distribution.
- Characteristics:
	- Appears as small random intensity fluctuations.
  - Commonly introduced by electronic sensor noise or poor lighting.
Example: Grainy texture in low-light photographs.
#### Salt-and-Pepper Noise:
- Description: Appears as bright (salt) and dark (pepper) pixels scattered randomly across the image.
- Cause: Bit errors in transmission or malfunctioning pixel sensors.
- Characteristics:
	- Isolated white and black dots.
	- Easy to detect but harder to correct without blurring.
#### Speckle Noise:
- Description: A granular noise pattern, often seen in medical ultrasound or radar images.
- Cause: Interference of coherent waves from reflective surfaces.
- Characteristics:
	- Forms a multiplicative noise (dependent on pixel intensity).
	- Reduces contrast and sharpness.
#### Poisson Noise (Shot Noise):
- Description: Variance in pixel intensities caused by the random nature of photon arrival at image sensors.
- Cause: Low-light conditions during image capture.
- Characteristics:
	- Intensity-dependent noise.
	- Prominent in scientific imaging.

## Task Definitions
The task involves implementing and evaluating methods to remove noise from images while preserving their details. The focus is on achieving high-quality outputs suitable for applications such as enhancing low-quality photos or processing medical images. Below are the detailed task definitions:
- Objective: Develop a robust, modular, and concurrent system for denoising images using multiple techniques, while preserving fine details and edges.

Some tasks that we can use to avoid the noise on an image:
1. Gaussian Filtering.
	- Pourpose: Smoothens images by averaging pixel values with their neighbors, reducing random noise.
	- Challenges: Avoiding excessive blurring of edges
 	- Parameters: Kernel size, standard deviation (sigma).
2. Median Filtering.
   	- Pourpose: Effectively removes salt-and-pepper noise by replacing pixel values with the median of their neighbors.
   	- Challenges: Maintaining computational efficiency for large kernel sizes.
   	- Parameters: Kernel size.
3. Wavelet Transforms.
   	- Pourpose: Removes noise in the frequency domain while preserving image details.
   	- Challenges: Choosing appropriate wavelet type, decomposition level, and thresholding strategy.
   	- Parameters: Wavelet type, thresholding mode (soft/hard), decomposition level.
