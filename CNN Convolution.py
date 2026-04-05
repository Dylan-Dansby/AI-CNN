import numpy as np
import imageio.v2 as io
import matplotlib.pyplot as plt # Corrected import from matplotlib.image to matplotlib.pyplot
import matplotlib as mtplt
from IPython.display import display, Image


image = Image('/content/image.jpg') # Corrected path
display(image)

imgArr = io.imread('/content/image.jpg')
print(len(imgArr))
print(imgArr.shape)
print(imgArr)


convFilter = imgArr[1:4, 1:4]
print('Shape of convFilter:', convFilter.shape)
print('Content of convFilter:\n', convFilter)

def convolve2d(image,kernel):
  image_height, image_width = image.shape
  kernel_height, kernel_width = kernel.shape

  output_height = image_height - kernel_height + 1
  output_width = image_width - kernel_width + 1

  stride = 2

  output_image = np.zeros((output_height, output_width), dtype=np.float32)
  for i in range(0, output_height, 2):
    for j in range(0, output_width ,2):
      roi = image[i:i + kernel_height, j:j + kernel_width]

      output_image[i,j] = np.sum(roi * kernel)

  return np.clip(output_image, 0, 255).astype(np.uint8)

new_image = convolve2d(imgArr, convFilter)
print('Shape of new_image:', new_image.shape)

# plt.imread(new_image) # This line is incorrect, plt.imread expects a file path
# plt.show(new_image) # This line is incorrect, plt.show expects a plot, not an image array



plt.figure(figsize=(6, 6))
plt.imshow(new_image, cmap='gray')
plt.title('Convolved Image')
plt.axis('off')
plt.show()

#MAX POOLING SECTION
mask = np.zeros((2, 2), dtype=np.float32)
print('Shape of mask:', mask.shape)
def max_pool(image, mask):
    image_height, image_width = image.shape
    mask_height, mask_width = mask.shape

    output_height = image_height
    output_width = image_width

    output_image = np.zeros((output_height, output_width), dtype=np.float32)
    for i in range(0, output_height, 2):
      for j in range(0, output_width ,2):
        roi = image[i:i + mask_height, j:j + mask_width]

        output_image[i,j] = np.max(roi)

    return np.clip(output_image, 0, 255).astype(np.uint8)

max_image = max_pool(new_image, mask)
print('Shape of max_image:', max_image.shape)

# downSz = (new_image.x() - 2)/2
plt.figure(figsize=(6, 6))
plt.imshow(max_image, cmap='gray')
plt.title('Max Pooled Image')
plt.axis('off')
plt.show()