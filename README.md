# Gait Recognition Algorithm in Multi-view Scene Based on Deep Learning
This project mainly shows how to perform gait recognition based on deep learning models. Gait recognition is a biometric identification method that uses human walking patterns to recognize individuals. This project mainly utilizes gait energy map (GEM), gait cycles, auto-encoders, and stacked progressive auto-encoders to perform gait recognition.

## Features
-   **Gait Energy Map (GEM) and Gait Cycle Extraction**
We need to extract gait cycles from a sequence of gai images. A gait cycle is a complete sequence of steps taken by an individual while walking. We can use the gait enegty map (GEM) to extract the cycles. The GEM is a binary map that highlights the areas of an image or video sequence that correspond to the moving parts of a person's body during walking. Once GEM is obtained, the gait cycle can be extracted by analyzing the peaks and valleys in the GEM.

-   **Auto-Encoder**
An auto-encoder can be used for feature extraction and data compression. The auto-encoder consists of an encoder network that maps the input data to a lower-dimensional latent space representation and a decoder network that reconstructs the input data from the latent space representation. In the context of gait recognition, an auto-encoder can be trained to extract features from the gait cycles.

-   **Stacked Progressive Auto-encoders (SPAs)**
A Stacked Progressive Auto-encoder (SPAE) is a type of auto-encoder that is trained in a layer-wise manner to learn hierarchical representations of the input data. In this project, an SPAE can be used to learn high-level representations of the gait cycles by stacking multiple layers of auto-encoders.


## Pipeline
The pipeline of this gait recognition project is designed as the following steps:

-   Prepocess the images and generate the gait energy map
-   Extract gait cycles by analyzing peaks and valleys in the map
-   Train the auto-encoder and stacked progessive auto-encoder. The input of the auto-encoder is the gait cycles, and the output is the reconstructed cycles.
-   Use classifier to recognize the gait

## Note
Due to copy right and privacy protection policies, I will not provide any information relate to the dataset.
