# Using Generated Adversarial Networks to Improve Computer Generated Image Quality

Combining fully computer generated images as part of a GAN input to be able to generate additional synthetic data, for use further down an image-classification pipeline. Strongly based off of [Apple's 2017 Paper "Improving the Realism of Synthetic Images"](https://machinelearning.apple.com/research/gan)

## GAN Design

Initially, I made a generator based off of a more traditional generator which takes a random vector as a seed instead of a generated image. This generator is based off of `Unsupervised Representation Learning With Deep Convolutional Generated Adversarial Networks` available [on Arxiv](https://arxiv.org/abs/1511.06434). This was modified by replacing the random network with a convolutional neural network, turning it into an auto decoder.

The discriminator was based off of one described on [Keras's blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).

Following the author's lead, I do some initial pretraining of the generator with a simple pass through and the L1-norm as the error function. 2000 prefitting iterations were chosen, which is not enough to reach convergence, but good enough to start the GAN's actual training.

![Training in prefitting mode](https://github.com/ToddBodnar/appl-eye-gan/raw/master/images/prefitting.gif)

Training in prefitting mode.

The actual generator is trained based off of two metrics: the L1-norm of the input image and it's ability to generate images that the discriminator classifies as 'real.' The training rate iterations based on the second metric is varied between runs to weight the two metrics differently.

![Training in adversarial mode](https://github.com/ToddBodnar/appl-eye-gan/raw/master/images/fitted_normal_more_passthrough_fitting.gif)

Training in adversarial mode.

The discriminator is trained by a comparison to real images and images generated by the current iteration of the generator.

## Training Data

All captured images were roughly centered and cropped to a 64x64 square with color. During training, Keras's ImageDataGenerator was used to introduce noise into the images (shear, translation, etc.)

### Real World Data

![](https://github.com/ToddBodnar/appl-eye-gan/raw/master/training_eye/eye1/scene00001.png) ![](https://github.com/ToddBodnar/appl-eye-gan/raw/master/training_eye/eye2/scene00071.png) ![](https://github.com/ToddBodnar/appl-eye-gan/raw/master/training_eye/eye3/scene00321.png)

A total of 729 images were collected from 3 videos of my eye with various lighting conditions, gazes, and expressions using my laptop's camera. Does this cause the final model to be overfitted because of the lack of diversity in eye shapes? Probably, but this data would be difficult to collect, and the final results seem good enough.

### Generated

![](https://github.com/ToddBodnar/appl-eye-gan/blob/master/training_generated/run1/scene00106.png) ![](https://github.com/ToddBodnar/appl-eye-gan/blob/master/training_generated/run1/scene00136.png) ![](https://github.com/ToddBodnar/appl-eye-gan/blob/master/training_generated/run1/scene00326.png)

118 images were captured from the workspace and demo model from [Crazy Talk 8](https://www.reallusion.com/crazytalk/download.html).

## Results

![Input data](https://github.com/ToddBodnar/appl-eye-gan/raw/master/training_generated/run1/scene00081.png) ![generated eye](https://github.com/ToddBodnar/appl-eye-gan/raw/master/images/final_adjusted_normal_more_passthrough_00011.jpg) ![A Real Eye](https://github.com/ToddBodnar/appl-eye-gan/raw/master/training_eye/eye2/scene00316.png)
