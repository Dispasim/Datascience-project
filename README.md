# Machine Learning Regression Model Report

## Abstract
In this report, I will outline the process through which I utilized deep learning techniques to develop a machine learning algorithm capable of performing regression. The model combines two distinct sets of data: one consisting of provided information, and the other derived from an audio file. By integrating these two data sources, the model aims to predict the target variable effectively.

## Problem Overview
This project involves working with a dataset containing information about voice recordings, such as the identity of the recorder, metadata related to the recording, and the path to the corresponding WAV file. The objective is to train a machine learning model that, given the recording information (excluding the age), can predict the age of the individual who made the recording.

An important consideration is that all recordings are compatible with the Librosa library, which will be pivotal. The way we handle and process the audio data could significantly enhance the model's performance.

Upon analyzing the distribution of the recorders' ages, we observe that the majority are relatively young, with most falling within the 15-25 age range. While there are some older individuals, the dataset contains only 109 individuals aged between 60 and 90.



## Proposed Approach

### Preprocessing

#### The Dataset
For the dataset, I opted to retain most rows and columns, as the dataset is already relatively clean. However, I made the following modifications:
- Removed the "Sampling rate" column because it contains a single constant value, which would not contribute useful information and could introduce noise into the dataset.
- Excluded the "Ethnicity" column to avoid the high dimensionality that would result from one-hot encoding, as this could negatively impact the model's performance.
- Dropped the "path" and "id" columns since they are identifiers and provide no relevant features for the analysis.
- Converted the "tempo" column from an array format to a float value, ensuring it is suitable for numerical processing.
- Used one hot encoding on the gender column.

These adjustments were made to streamline the dataset and focus on meaningful features for the model.

#### The Audio Files
To preprocess the audio files, I converted them into mel spectrograms, as this representation is required for the chosen model. Although I initially experimented with data augmentation techniques on the spectrograms, the results deteriorated, leading me to exclude augmentation from the pipeline.

Additionally, I applied MinMax normalization to both the input data and the spectrogram images to standardize their scales. For the target values, I employed a logarithmic transformation to reduce skewness and achieve a more uniform distribution, which could improve the model's performance.

### Model Selection
In previous coursework, I acquired foundational knowledge in deep learning, with a particular focus on convolutional neural networks (CNNs) for image classification tasks. Building on this foundation, my approach involves leveraging deep learning techniques to extract relevant features from audio files, which are then combined with the provided data to perform regression using neural networks.

Through further research, I identified a study that outlines a methodology for this process. The approach utilizes the Librosa library to transform audio files into mel spectrograms, followed by the application of a CNN for feature extraction. So I chose to implement this approach and find a way to introduce the dataset into the network so the algorithm can make the best use of these two sources of information.

### Hyperparameters Tuning
The first step of the neural network is to convert the images from 128x128 images to a vector while keeping the maximum amount of information possible.

![First half of the neural network](path/to/your/image.png)

For that, we alternate operations of convolution and maxpooling. 

The convolution operation in a CNN applies a small filter (kernel) over an input (e.g., image) to extract features like edges or patterns by performing element-wise multiplications and summing the results, producing a feature map. This process reduces parameters, provides translation invariance, and captures spatial features efficiently.

![Convolution operation](path/to/your/image.png)

Max-pooling is a downsampling operation used in CNNs to reduce the spatial dimensions of feature maps while retaining important features. It divides the input into non-overlapping regions (e.g., 2×2), takes the maximum value from each region, and produces a smaller feature map. This helps reduce computational cost, control overfitting, and focus on dominant features such as edges or textures.

![Maxpooling operation](path/to/your/image.png)

Flatten is a layer in neural networks that converts a multi-dimensional input (e.g., a 2D feature map from convolutional layers) into a 1D vector. This transformation is necessary to connect convolutional layers to fully connected layers for classification or regression tasks.

Now that we have a vector representing the image, we can pass it to a neural network to do regression on it.

![Second half of the neural network](path/to/your/image.png)

However, it is essential to incorporate the data from the dataset. After conducting several experiments, I opted for an alternative neural network that transforms the 15 dimensions into a 32-dimensional vector, which is then aggregated before the final step. This approach ensures that the dataset is assigned the appropriate level of importance in the model.

## Results
The results are rather disappointing. After trying different techniques and configurations, I was unable to get better than 9.7 on my test dataset and 10 on the evaluation dataset.

## Discussion
As I said, I’m disappointed by the results. I am sure that my approach can be good and that my result can be greatly improved, but I did not manage to do it.

Also, in the article from which I took inspiration, the author used pretrained algorithms from PyTorch like EfficientNet, SE-ResNext, or NFNet. Maybe it is a better approach than a basic CNN as I did?

## References
[1] Article deep learning.
