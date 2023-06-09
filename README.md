# Brain MRI Tumor Classification Deep Learning Model

This is a deep learning model that can classify MRI images of the brain into four categories: glioma tumor, meningioma tumor, no tumor, and pituitary tumor. The model was trained on the Images Dataset "Brain Tumor Classification (MRI)" From Kaggle by SARTAJ under the CC0: Public Domain License.

Trained Model File: https://huggingface.co/ShabGaming/Brain_MRI_Tumor_Classification

## Model
The model is a convolutional neural network (CNN) with the following architecture:
```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 1248, 1248, 32)    896
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 624, 624, 32)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 622, 622, 64)      18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 311, 311, 64)      0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 309, 309, 128)     73856
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 154, 154, 128)     0
_________________________________________________________________
flatten (Flatten)            (None, 307328)            0
_________________________________________________________________
dense (Dense)                (None, 128)               39338112
_________________________________________________________________
dropout (Dropout)            (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 516
=================================================================
Total params: 39,436,876
Trainable params: 39,436,876
Non-trainable params: 0
```
The model was trained using TensorFlow and achieved an accuracy of over 95% on the validation set.

## GUI
In addition to the model, we have also provided a graphical user interface (GUI) that allows users to upload an MRI image and get a prediction from the model. The GUI was built using the Tkinter library in Python.

To use the GUI, simply run the gui.py file and a window will appear. Click the "Choose File" button to select an MRI image from your computer, and then click the "Predict" button to get the model's prediction. The GUI will display the selected image as well as the predicted class.

## Usage
To use the model and GUI, follow these steps:
- Clone or download the GitHub repository containing the model and GUI files.
- Install the necessary Python libraries.
- Train the model by running 'BrainTumorMRIDetection.ipynb'. This will save the trained model as a .h5 file in the repository directory (You can also just download the model, more information down below).
- Run the GUI by running gui.py. This will open a window where you can upload an MRI image and get a prediction from the model.

## Credits
Muhammad Shahab Hasan (Shab)
- https://www.fiverr.com/best_output
- https://www.youtube.com/Shabpassiongamer
- https://medium.com/@ShahabH

The "Brain Tumor Classification (MRI)" dataset was created by SARTAJ and is licensed under the CC0: Public Domain License.
