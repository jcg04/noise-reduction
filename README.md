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
