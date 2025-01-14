# Noise-reduction
Remove noise from images while preserving details.

## Content
1. What are noise in images?
2. Task Definitions.
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
   
   A Gaussian Filter is a low pass filter used for reducing noise (high frequency components) and blurring regions of an image. The filter is implemented as an Odd sized Symmetric Kernel (DIP version of a 	 
   Matrix) which is passed through each pixel of the Region of Interest to get the desired effect. The kernel is not hard towards drastic color changed (edges) due to it the pixels towards the center of the 	 
   kernel having more weightage towards the final value then the periphery. A Gaussian Filter could be considered as an approximation of the Gaussian Function (mathematics). In this article we will learn 
   methods of utilizing Gaussian Filter to reduce noise in images using Python programming language.
	- Pourpose: Smoothens images by averaging pixel values with their neighbors, reducing random noise.
	- Challenges: Avoiding excessive blurring of edges
 	- Parameters: Kernel size, standard deviation (sigma).
2. Median Filtering.
	
	For this we have to understand all the High-Pass filter and the Low-Pass filter. The High-Pass filter remove the high frequency domain, while the Low-Pass filter eliminates low-frequency regions while 
        retaining or enhancing the frequency components.
   	- Pourpose: Effectively removes salt-and-pepper noise by replacing pixel values with the median of their neighbors.
   	- Challenges: Maintaining computational efficiency for large kernel sizes.
   	- Parameters: Kernel size.
4. Wavelet Transforms.
   
   	There are several types of Wavelet transforms, each suitable for different applications. The Continuous Wavelet Transform (CWT) provides a continuous 	 
        representation of the signal, allowing a detailed analysis of its structure. In contrast, the Discrete Wavelet Transform (DWT) offers a more 
        computationally efficient approach by discretising both the time and frequency domains. The DWT is widely used in applications such as image compression 
        and noise reduction, as it allows the efficient representation of data while preserving essential features. In addition, the Stationary Wavelet Transform 
        (SWT) maintains the length of the original signal, which makes it useful for applications where phase information is critical.
   
   	- Pourpose: Removes noise in the frequency domain while preserving image details.
   	- Challenges: Choosing appropriate wavelet type, decomposition level, and thresholding strategy.
   	- Parameters: Wavelet type, thresholding mode (soft/hard), decomposition level.

## Solution Key Ideas:
To effectively remove noise from images while preserving details, the solution should be based on a combination of robust algorithms, efficient implementation, and scalable design. Below are key ideas that can enhance the solution beyond the basic techniques:
1. Adaptative Techniques:
Noise in images can vary in type and intensity, so adaptive methods can dynamically adjust based on the image characteristics:
	- Adaptive Gaussian Filtering: Automatically adjust kernel size and sigma based on local image properties, such as edge density.
	- Adaptive Median Filtering: Use smaller kernels in smooth regions and larger kernels in noisy regions.
	- Wavelet-Based Adaptive Thresholding: Set thresholds based on the noise level in each sub-band for better preservation of details.
2. Hybrid Approaches:
Combining techniques can yield better results:
	- Combination of Gaussian and Median Filters: Use Gaussian smoothing for overall noise reduction and Median filtering for handling salt-and-pepper noise.
	- Preprocessing with Denoising Autoencoders: Use deep learning models to preprocess images before applying traditional filters.
	- Wavelet + Non-Local Means (NLM): Use wavelet transforms to denoise high-frequency components and NLM for smoothing textures.
3. Edge-Preserving Filters:
Preserving edges is crucial for high-quality image restoration:
	- Bilateral Filter: A spatial filter that smoothens noise while preserving edges by considering both spatial and intensity distances.
	- Non-Local Means (NLM): Reduces noise by averaging similar patches in the image, preserving textures and edges.
	- Anisotropic Diffusion: Reduces noise iteratively while ensuring edge preservation by smoothing within regions rather than across edges.
4. Deep-Learning Based Denoising:
Modern approaches leverage the power of neural networks:
	- Convolutional Neural Networks (CNNs):
		- Train models like DnCNN (Denoising CNN) for different types of noise.
		- Advantages: Ability to learn noise patterns and adapt across varying types of noise.
	- UNet Architectures:
		- Perform pixel-wise noise reduction, ideal for preserving high-frequency details.
	- Denoising Autoencoders:
		- Learn a mapping from noisy to clean images in an unsupervised manner.
	- Transformers:
		- Emerging models in vision tasks can be adapted for denoising by leveraging global context information.
5. Parallel and Batch Processing:
To make the solution scalable:
	- Parallel Processing:
		- Use multithreading or multiprocessing to handle large datasets or high-resolution images efficiently.
	- Batch Denoising:
		- Process multiple images concurrently using GPUs or distributed systems.

## Implementation
We are going to implement ways to solve noise problems or to reduce them. We will implement it in python so we will need to download the numpy, matplotlib and pillow libraries.
For example to install the last two:
```
pip install Pillow matplotlib
```
I have implemented both gaussian filter, median filter and wavelet. Let's start with the gaussian filter:
```
 ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
 gaussian = np.exp(-0.5 * (ax / sigma) ** 2)
 kernel_1d = gaussian / gaussian.sum()
 kernel_2d = np.outer(kernel_1d, kernel_1d)
```
- ax: Creates a symmetric range of values centred on 0 (e.g. for kernel_size=5, [-2, -1, 0, 1, 2] is generated).
- gaussian: Calculate the Gaussian function in 1D for the range ax.
- kernel_1d: Normalises the Gaussian values by dividing each value by the total sum. This ensures that the kernel sum is 1.
- kernel_2d: Generates a two-dimensional (2D) kernel by the outer product of kernel_1d with itself. This simulates the 2D form of the Gaussian function.

```
 padded_image = np.pad(image, kernel_size // 2, mode='edge')
 filtered_image = np.zeros_like(image, dtype=np.float32)
```
- padded_image: Enlarges the original image by adding reflected edges. This ensures that pixels close to the edges of the original image can be processed correctly.
- filtered_image: Initialises an empty array with the same dimensions as the original image. This is where the filtered values will be stored.

```
 for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.sum(region * kernel_2d)
```
- region: Extracts a submatrix (window) from the zoomed image. The window has the same size as the kernel (kernel_size x kernel_size) and is centred on the current pixel.
- Multiplication and Addition: Performs the element-by-element product between the region window and the kernel kernel_2d.
It then sums all product values and assigns the result to the corresponding pixel in filtered_image.

Let's go now with the median filter:
```
if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
```
Ensures that the kernel size is odd.
This is necessary because the median is defined as the central value of an ordered set. With an even size, there would be no clear centre.

```
pad_size = kernel_size // 2
padded_image = np.pad(image, pad_size, mode='edge')
filtered_image = np.zeros_like(image, dtype=np.uint8)
```
- pad_size: Calculates the number of additional rows and columns needed to handle the edges of the image.
- padded_image: Enlarges the image with edge-reflected pixels (mode=‘edge’). This avoids problems when processing pixels close to the edges.
- filtered_image: Initialises an empty array to store the results after applying the filter. It has the same size as the original image.

```
for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extraer la región alrededor del píxel actual
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Calcular la mediana de la región
            filtered_image[i, j] = np.median(region)
```
- region: Extracts a window of size kernel_size x kernel_size centred on the current pixel. This window includes the neighbouring pixels of the zoomed image.
- Calculate median: Sorts all values in the extracted region. Finds the central value of the sorted set. This value replaces the original pixel value at the corresponding position.

And finally the wavelet transform:
```
 h, w = image.shape
    half_h, half_w = h // 2, w // 2
```
- h, w: Height and width of the image.
- half_h, half_w: Half of the height and width, to calculate the lower resolution sub-bands (approximation and detail).

```
 LL = (image[0:h:2, 0:w:2] + image[1:h:2, 0:w:2] + 
          image[0:h:2, 1:w:2] + image[1:h:2, 1:w:2]) / 4

    LH = (image[0:h:2, 0:w:2] + image[1:h:2, 0:w:2] - 
          image[0:h:2, 1:w:2] - image[1:h:2, 1:w:2]) / 4

    HL = (image[0:h:2, 0:w:2] - image[1:h:2, 0:w:2] + 
          image[0:h:2, 1:w:2] - image[1:h:2, 1:w:2]) / 4

    HH = (image[0:h:2, 0:w:2] - image[1:h:2, 0:w:2] - 
          image[0:h:2, 1:w:2] + image[1:h:2, 1:w:2]) / 4
```
- LL (Low Frequency):This is the approximation sub-band. It averages 4 neighbouring pixels (pairs) to reduce resolution and capture overall image trends.
- LH (Vertical High Frequency): Detects vertical changes in pixels.
- HL (Horizontal High Frequency): Detects horizontal changes in pixels.
- HH (High Frequency Diagonal): Captures details and diagonal variations in the image.

```
 LH = np.where(np.abs(LH) > threshold, LH, 0)
 HL = np.where(np.abs(HL) > threshold, HL, 0)
 HH = np.where(np.abs(HH) > threshold, HH, 0)
```
For high frequency sub-bands (LH, HL, HH):
If an absolute value is less than threshold, it is removed (set to 0).
This reduces the high-frequency components that represent noise, while keeping the important details.

```
  result = np.zeros_like(image, dtype=np.float32)
  for i in range(half_h):
     for j in range(half_w):
            result[2 * i, 2 * j] = LL[i, j] + LH[i, j] + HL[i, j] + HH[i, j]
            result[2 * i + 1, 2 * j] = LL[i, j] + LH[i, j] - HL[i, j] - HH[i, j]
            result[2 * i, 2 * j + 1] = LL[i, j] - LH[i, j] + HL[i, j] - HH[i, j]
            result[2 * i + 1, 2 * j + 1] = LL[i, j] - LH[i, j] - HL[i, j] + HH[i, j]
```
The reconstructed pixel combines the low and high frequency sub-bands:
It is calculated using the inverse Haar decomposition ratios.
The loops traverse the dimensions of the (lower resolution) sub-bands and reconstruct into the full resolution image.

[1] https://www.geeksforgeeks.org/spatial-filters-averaging-filter-and-median-filter-in-image-processing/?ref=gcse_outind
[2] https://www.geeksforgeeks.org/apply-a-gauss-filter-to-an-image-with-python/?ref=gcse_outind
