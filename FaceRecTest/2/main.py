import cv2
import numpy as np

# Define the image path
orgimage = cv2.imread("tree.jpg")
# Demensions for resized image
width = 500
height = int(orgimage.shape[0] * (width / orgimage.shape[1]))
# Define the image resizing
image = cv2.resize(orgimage, (width, height))
# Apply effects to the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
median_filtered_image = cv2.medianBlur(gray_image, 5)
edges = cv2.Canny(gray_image, 100, 200)
# Noise addition and denoising
noise = np.zeros(gray_image.shape, np.uint8)
cv2.randn(noise, (0), (25))
noisy_image = cv2.add(gray_image, noise)
denoised_image = cv2.fastNlMeansDenoising(noisy_image, None, 10, 7, 21)

cv2.imshow("Original Image", image)
cv2.imshow("Resized Image", image)
cv2.imshow("Grayscale Image", gray_image)
cv2.imshow("Gaussian Blur", blurred_image)
cv2.imshow("Median Filter", median_filtered_image)
cv2.imshow("Edges", edges)
cv2.imwrite("gray_image.jpg", gray_image)
cv2.imwrite("blurred_image.jpg", blurred_image)
cv2.imwrite("median_filtered_image.jpg", median_filtered_image)
cv2.imwrite("edges.jpg", edges)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Denoised Image", denoised_image)
cv2.imwrite("noisy_image.jpg", noisy_image)
cv2.imwrite("denoised_image.jpg", denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()