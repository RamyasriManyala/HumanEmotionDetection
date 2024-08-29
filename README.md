# Sentiment Analysis through Facial Expression Recognition Using Machine Learning

## Project Overview ##

This project focuses on the development of a facial expression recognition system using Convolutional Neural Networks (CNNs), specifically leveraging the VGG16 architecture. The model is trained to detect and classify seven different facial expressions: anger, disgust, fear, happiness, sadness, surprise, and neutrality. The system is designed with real-time applications in mind, particularly in the domains of mental health monitoring, customer service, and human-computer interaction.

## Features ##

Model Architecture: VGG16 CNN, fine-tuned for facial expression recognition.
Dataset: The model was trained and evaluated on the FER-2013 dataset, which includes over 35,000 labeled images.
Transfer Learning: The model utilizes transfer learning by fine-tuning a pre-trained VGG16 model, originally trained on ImageNet, to adapt it for emotion detection.
Real-Time Detection: The system is capable of real-time facial expression recognition, making it suitable for applications in dynamic environments.
Visualization Tools: Includes scripts for visualizing model performance metrics, confusion matrix, and PCA plots.

Installation

##  To clone and run this project, you'll need Git and Python installed on your machine. ##

``` bash
# Clone the repository
https://github.com/RamyasriManyala/HumanEmotionDetection.git

```

## Usage ##

- Training the Model
To train the model, execute the following command. Ensure you have the FER-2013 dataset downloaded and placed in the appropriate directory.

```bash

python train_model.py --dataset /path/to/fer2013

```
- Evaluating the Model

To evaluate the model's performance on the test dataset:

```bash

python evaluate_model.py --model /path/to/saved_model.h5 --dataset /path/to/fer2013/test

```

- Real-Time Emotion Detection
For real-time emotion detection using your webcam:
```bash

python real_time_detection.py --model /path/to/saved_model.h5
```
- Visualization
To visualize the training process, including accuracy and loss curves, confusion matrix, and PCA of the dataset:

```bash

python visualize_results.py --model /path/to/saved_model.h5 --dataset /path/to/fer2013/test

```
### Directory Structure ###
```bash

.
├── README.md
├── train_model.py
├── evaluate_model.py
├── real_time_detection.py
├── visualize_results.py
├── data/
│   ├── train/
│   ├── test/
│   └── fer2013.csv
├── models/
│   ├── vgg16_finetuned.h5
├── logs/
│   ├── training_log.csv
└── requirements.txt

```
## Model  ##

The model achieved an accuracy of approximately 55% on the test dataset, with higher precision in detecting distinct emotions such as happiness and surprise. However, the model showed challenges in distinguishing between emotions like sadness and neutrality. The confusion matrix and other performance metrics are available in the visualization results.

 ## Future Work ##

### Future improvements could include: ###

Expanding the dataset to include more diverse facial expressions and demographics.
Implementing advanced architectures such as Capsule Networks to improve subtle emotion detection.
Integrating additional modalities such as voice or physiological signals for multimodal emotion recognition.
License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgements ###

Dataset: FER-2013 Dataset
Pre-trained Model: VGG16 weights from ImageNet
