# Bird Classification via Transfer learning and using PyTorch Lightning
Small Project to learn a bit more about TorchVision and Pre-trained models using the 525 Bird Classifcation Dataset on Kaggle (https://www.kaggle.com/datasets/gpiosenka/100-bird-species/code). 
The notebook and associated files are meant to be run in Google Colab.

Currently only using ResNet18 but would be fun to try other pre-trained models and changing the interior layers instead of just the output layer to see if results could be improved. ResNet34 already performs very well by itself so it would be fun to try to improve a pre-trained model which performs less optimally.

This small project uses the available pre-trained ResNet18 Model for image classification with the final layer being replaced to fit the number of bird classifcation classes. Furthermore, none of the original ResNet18 Layer weights were optimised during training with only the last layer having it's weight's optimsed to see how transfer learning could be used. PyTorch Lightning was used on-top of Pytorch for training and testing.

The trained model attained a precision and accuracy of 90% and could probably attain +95% if fine-tuned further.
Very Nice! We like!