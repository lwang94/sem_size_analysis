# Brief Overview
------------------
SAEMI is a tool that uses deep learning to obtain the size distribution of nanoparticles in electron microscopy images.
The deep learning model is a [unet](https://arxiv.org/pdf/1505.04597.pdf) with a resnet18 downsample created 
through the [fastai](https://github.com/fastai/fastai) library. See below for a brief outline of the computation
process:

Upload Image &rarr; Preprocess Image &rarr; Segment Image Using U-net Model &rarr; Give Unique Labels to Each Segment and Count Them &rarr; Display Histogram & Segmented Image

# Upload Image
---------------------
To upload an image, use the <b> UPLOAD IMAGE </b> at the top of the app. To ensure that the best results are obtained for this app, 
please ensure that the image has minimal noise and imaging artifacts. Watermarks and additional meta information (such as the scale bar) may also
be mislabeled as a particle during the segmentation process and so should be removed before uploading (although mislabeled segement can be removed
post segmentation; see Display Histogram and Segmented Image below). Tools to minimize noise or remove watermarks include Python based libraries 
such as [scikit-image](https://scikit-image.org/) and [opencv](https://opencv.org/) or for a more GUI based approach, [ImageJ](https://imagej.net/Welcome).

# Preprocess Image
----------------------
Before the model can segment the image, it first needs to have dimensions of 192x256x3 where 192 is the pixel height, 256 is the pixel width, and 3 is the number of channels 
referring to RGB color values. If the image has only one channel, it will be repeatedly stacked three times along the channel axis to obtain a 3 channel image. More importantly,
the image height and width will automatically be resized to 192x256 pixels using a bilinear interpolation method. The bilinear interpolation is done using <a href="https://scikit-image.org/docs/dev/api/skimage.transform.html?ref=driverlayer.com/web#skimage.transform.resize">
skimage.transform.resize</a>.

# Segment Image Using U-net Model
------------------------
The model used to segment the image was trained using SEM images obtained from [NFFA-EUROPE](https://b2share.eudat.eu/records/80df8606fcdb4b2bae1656f0dc6db8ba). The model was determined
to have 97% accuracy using the [Dice metric](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2). The result of segmentation is a 192x256
binary image consisting of 0s and 1s where 0 represents a background pixel and 1 represents a particle pixel. 

# Give Unique Labels to Each Segment and Count Them
-----------------------------------
To obtain a size distribution, the resulting binary image is further processed. Connected regions of 1s in the binary image is given a unique label eg.
```
[[1, 0, 0, 1, 1],      [[1, 0, 0, 2, 2],
 [0, 0, 0, 1, 1],       [0, 0, 0, 2, 2],
 [1, 1, 0, 0, 0],  -->  [3, 3, 0, 0, 0],
 [1, 0, 0, 0, 0],       [3, 0, 0, 0, 0],
 [0, 0, 0, 1, 1]]       [0, 0, 0, 4, 4]]
```
The number of unique labels are then counted up and put into a separate array to represent the sizes of the particles in the image. For the example above,
the size distribution would be:
```
{1:1, 2:4, 3:3, 4:2} --> [1, 4, 3, 2].
```
Since the "pixel size" is calculated for the 192x256 binary image, the elements in the array are then multiplied by a rescaling factor to match the size of the
original image. For instance, if the above list was obtained for an image that was originally 768x1024 pixels in size, the rescaling factor is:
```
(768x1024) / (192x256) = 16
```
and the final array would be:
```
[16, 64, 48, 32].
```
<b>Please note that the "sizes" that are calculated is the number of pixels that a particle takes up in the image and it is left up to the user to convert from pixels
to a physical size</b>. 

# Display Histogram & Segmented Image
----------------------------
Finally, a histogram of the size distribution is displayed with the Mean, Median, and Standard Deviation of sizes already calculated. The size distribution list can be downloaded by clicking
on <b>Download Data</b>. Below the histogram, the uniquely labeled segmented binary image is shown on the left and on the right is the segmented binary image overlayed with the original image so 
the user can determine the success of the segmentation. 

Users can click on uniquely labeled segments (particles) in the left image to remove them from the size distribution. This is to account for any potentially mislabeled segments that introduce 
error into the size distribution. The slider below the right image can be used to change the opacity of the segmented binary image so that users can choose to see more of the original image
or more of the segmented binary image.