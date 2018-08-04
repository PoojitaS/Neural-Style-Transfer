# STYLE WEIGHT OPTIMIZATION FOR MULTI-STYLE TRANSFER 

Repository for documenting ongoing developments in research dedicated to advancing Neural Style Transfer methods.

## OVERVIEW
### Neural Style Transfer
In 2015, Gatys et al. published a paper titled ['A Neural Algorithm of Artistic Style'](https://arxiv.org/pdf/1508.06576v2.pdf). 
It proposed an optimization-based method to transfer artistic style from a painting onto a photograph such that the resulting pastiche 
exhibits a blend of style of the painting and the content of the photograph. 

#### How they did it
The main idea is to construct an optimization problem that minimizes loss of information (in the form of style and content) between 
input images (content image and style image) and generated image (pastiche). A pre-trained (on image dataset) deep convolutional neural network (VGG) 
is used to backpropogate into the generated image to match feature representations with the input images. This process takes several iterations 
to produce effective results, and is often very slow. 

### Multi-Style Transfer
Extending from the Neural Style Transfer concept, we can blend multiple styles onto one content image to generate a beautiful pastiche.
This can be done by assigning style weights to each painting to fix each painting's contribution to the blended style. 

#### Optimizing style weights
Blending styles is often a tedious process involving many trial-and-error rounds for adjusting style weights in order to find the 
most asthetically pleasing image. In this project, I wrap an optimizer around the Gatys implementation of NST for multi-style transfer 
that finds optimal style weights that minimze the loss. 

## HOW TO RUN 
### Pre-requisites
This program has been implemented in Python 2.7. The following libraries must be installed on your system to run it:
* Numpy
* Scipy
* Keras
* h5py
* Tensorflow

### Execution
Clone this repository to your preferred directory on your system  (GPU preferred for faster performance)
Run the following command from your terminal within the directory: 
<pre> python optimize_nst.py path/to/content/image path/to/content/image path/to/style/image(s) path/to/generated/image</pre>
Example:
<pre>python optimize_nst.py images/inputs/content/Dipping-Sun.jpg images/inputs/style/the_scream.jpg images/inputs/style/wave_kanagawa.jpg Results/generated</pre>

## ACKNOWLEDGEMENT
The Gatys implementation for this project in Keras was borrowed from the following Github repository: [titu1994/Neural-Style-Transfer](https://github.com/titu1994/Neural-Style-Transfer).

## ABOUT AUTHOR
I'm currently studying MS in Computer Science from Georgia Institute of Technology. This project was part of my Summer Research Work under 
the guidance of Prof. Tucker Balch. Significant insights were provided by my fellow team member Keshav Ramani. 

