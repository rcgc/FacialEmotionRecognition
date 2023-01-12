# Facial Emotion Recognition

## Abstract
Human behaviour is a complex case study, especially when performing emotion recognition: Gestures that correspond to an emotion for a given face could not be accurate enough when predicting another one. This project is a simple approach to accomplish such task by implementing a neural network in keras and tensorflow combined with an artificial vision library known as OpenCV.<br>

## Introduction
Emotion recognition is the process of identifying human emotion. People vary widely in their accuracy at recognizing the emotions of others. Use of technology to help people with emotion recognition is a relatively nascent research area. Generally, the technology works best if it uses multiple modalities in context.<br>

## Materials and methods
### Technology stack
| Aspect                  | Tool                |
| -------------           | -------------       |
| Hardware accelerator    | GPU                 |
| Framework               | Tensorflow          |
| Libraries               | Keras & OpenCV      |
| NN Type                 | Convolutional       |
| Architecture            | VGG11               |
| Programming language    | Python & JavaScript |
| Code editor             | Google Colab        |
| Web browser             | Google Chrome       |

Table 1. </b>Project tech stack<br>

### Dataset
The dataset used for this project was taken from Jonathan Gheix kaggle account[18]. In the following graph we can visualize the different emotions found: <br>

![Emotion_frequencies](https://github.com/rcgc/FacialEmotionRecognition/blob/master/readme_images/emotion_frequencies.png)
<p><b>Figure 3. </b>Emotion frequencies</p><br>

### Results
![Training_accuracy_vs_validation_accuracy](https://github.com/rcgc/FacialEmotionRecognition/blob/master/readme_images/trainAcc_vs_valAcc.png)
<p><b>Figure 4. </b>Training accuracy vs Validation accuracy</p><br>


![Training_loss_vs_validation_loss](https://github.com/rcgc/FacialEmotionRecognition/blob/master/readme_images/trainLoss_vs_valLoss.png)
<p><b>Figure 6. </b>Training loss vs Validation loss</p><br>

Due to early stopping training accuracy reached between 70-75% and validation accucary reached 60-66%.<br>

### How to use it

Run all the code snippets from the jupyter notebook from top to bottom and choose GPU as runtime type/hardware accelerator.<br>

![Application_usage](https://github.com/rcgc/FacialEmotionRecognition/blob/master/readme_images/emotion_recognition_usage.jpeg)
<p><b>Figure 5. </b>Application usage</p><br>

## Conclusion
This is not a complete facial emotion recognition system due to we had to reduce the emotions dataset from 7 to 4 in order to increase training and validation accuracy. Also, we not only need images to predict the emotion that some could be feeling in a especific moment, we need to detect many other aspects such as substances involved in the process like adrenaline, cortisol, etc. However, this information could be used to help patients to improve mental health by monitoring their humor during the day and keeping a record of their progress<br>

## References
[1]S. Russell and P. Norvig, Artificial intelligence, 4th ed. 2020.<br>
[2]Chollet, F., 2022. Deep Learning With Python. 2nd ed. Greenwich, USA: Manning Publications.<br>
[3]"Neural Networks: Chapter 6 - Neural Architectures", Chronicles of AI, 2022. [Online]. Available: https://chroniclesofai.com/neural-networks-chapter-6-neural-architectures/. [Accessed: 20- May- 2022].<br>
[4]K. Team, “Keras: the Python deep learning API,” Keras.io, 2022. https://keras.io/ (accessed May 27, 2022).<br>
[5]Wikipedia Contributors, “Keras,” Wikipedia, Apr. 05, 2022. https://en.wikipedia.org/wiki/Keras (accessed May 29, 2022).<br>
[6]Wikipedia Contributors, “PyTorch,” Wikipedia, May 23, 2022. https://en.wikipedia.org/wiki/PyTorch (accessed May 29, 2022).<br>
[7]“About - OpenCV,” OpenCV, Nov. 04, 2020. https://opencv.org/about/ (accessed May 29, 2022).<br>
[8]Wikipedia Contributors, “scikit-learn,” Wikipedia, Jan. 14, 2022. https://en.wikipedia.org/wiki/Scikit-learn (accessed May 29, 2022).<br>
[9]“Caffe | Deep Learning Framework,” Berkeleyvision.org, 2012. http://caffe.berkeleyvision.org/ (accessed May 29, 2022).<br>
[10]“TensorFlow,” TensorFlow, 2022. https://www.tensorflow.org/ (accessed May 27, 2022).<br>
[11]Wikipedia Contributors, “TensorFlow,” Wikipedia, May 01, 2022. https://en.wikipedia.org/wiki/TensorFlow (accessed May 27, 2022).<br>
[12]chrisbasoglu, “The Microsoft Cognitive Toolkit - Cognitive Toolkit - CNTK,” Microsoft.com, Feb. 16, 2022. https://docs.microsoft.com/en-us/cognitive-toolkit/ (accessed May 27, 2022).<br>
[13]C. de, “clase de las redes neuronales profundas, más comúnmente aplicada al análisis de imágenes visuales,” Wikipedia.org, Jun. 23, 2014. https://es.wikipedia.org/wiki/Red_neuronal_convolucional (accessed May 31, 2022).<br>
[14]M. Basavarajaiah, "6 basic things to know about Convolution", Medium, 2022. [Online]. Available: https://medium.com/@bdhuma/6-basic-things-to-know-about-convolution-daef5e1bc411. [Accessed: 06- Jun- 2022].<br>
[15]A. Kaushik, "Understanding the VGG19 Architecture", OpenGenus IQ: Computing Expertise & Legacy, 2022. [Online]. Available: https://iq.opengenus.org/vgg19-architecture/. [Accessed: 21- May- 2022].<br>
[16]R. Alake, "Deep Learning: GoogLeNet Explained", Medium, 2022. [Online]. Available: https://towardsdatascience.com/deep-learning-googlenet-explained-de8861c82765. [Accessed: 06- Jun- 2022].<br>
[17]P. Ruiz, "Understanding and visualizing ResNets", Medium, 2022. [Online]. Available: https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8. [Accessed: 06- Jun- 2022].<br>
[18]J. Oeix, “Face expression recognition dataset”. https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset<br>
[19]A. Sharma, "Emotion Detector using Keras - with source code - easiest way - easy implementation - 2022 - Machine Learning Projects", Machine Learning Projects, 2022. [Online]. Available: https://machinelearningprojects.net/emotion-detector-using-keras/. [Accessed: 13- Jun- 2022].<br>
[20]J. Kodithuwakku, D. Arachchi and J. Rajasekera, "An Emotion and Attention Recognition System to Classify the Level of Engagement to a Video Conversation by Participants in Real Time Using Machine Learning Models and Utilizing a Neural Accelerator Chip", Algorithms, vol. 15, no. 5, p. 150, 2022. Available: https://www.mdpi.com/1999-4893/15/5/150. [Accessed 13 June 2022].
