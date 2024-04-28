The use of deep learning algorithms has been a game-changer in the field of cardiovascular imaging, allowing for the extraction of
intricate features and patterns from ECG data, resulting in highly accurate classification of various cardiac conditions. This article
aims to explore the effectiveness of convolutional neural networks and recurrent neural networks with long short-term memory
layers in ECG classification, highlighting their ability to capture temporal dependencies and noise while offering interpretability of
hidden states. Additionally, we will discuss the advantages of deep learning approaches over traditional machine learning solutions,
which often rely on heuristic hand-crafted features or shallow learning architectures. Through an extensive analysis of more than
300 research articles in the medical field, it has been determined that deep learning techniques, particularly convolutional neural
networks and recurrent neural networks with LSTM layers, have shown great promise in the classification of ECG data. These deep
learning algorithms have the advantage of not requiring hand-crafted or engineered features, as they can automatically learn and
extract relevant features from the raw ECG data (Romiti et al., 2020). Furthermore, these deep learning techniques have
demonstrated superior performance compared to human results, showcasing their ability to accurately classify various cardiac
conditions without the need for manual intervention.
METHODOLOGIES:
A. CNN Model
The entire document should We used Deep Learning techniques such as CNN, RNN with LSTM layers and proposed two models
where first model is actually CNN and secondly we combined CNN with RNN. This model architecture starts with a Convolutional
Neural Network (CNN) component, consisting of three Conv2D layers. The first Conv2D layer has 32 filters with a filter size of 3x3
and utilizes the Rectified Linear Unit (ReLU) activation function. It takes input images with a shape of 150x150 pixels and 3 color
channels, representing RGB. Following the Conv2D layer, a MaxPooling2D layer with a pool size of 2x2 is applied to downsample
the feature maps. The figure 1 shows the CNN model architecture.
![image](https://github.com/naseer30/ECG-CLASSIFICATION-USING-DEEP-LEARNING/assets/103625760/e88431c8-98f2-4ba6-9089-91728edd6c00)
B. CNN-RNN Model
The figure 2 shows the architecture of proposed CRNN model. Our proposed model combines convolutional layers (Conv2D and
MaxPooling2D) for feature extraction with recurrent layers (LSTM) for sequential temporal data processing. While the LSTM
layers evaluate the sequential information produced by the convolutional layers, convolutional layers capture spatial features from
the input images. The model architecture integrates Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN)
components to process Electrocardiogram (ECG) images efficiently for classification. It begins with three Conv2D layers followed
by MaxPooling2D layers, progressively increasing the number of filters from 32 to 128. Each Conv2D layer uses a 3x3 filter size
and Rectified Linear Unit (ReLU) activation function, processing input images of 150x150 pixels and 3 color channels (RGB).
Subsequently, MaxPooling2D layers down sample the feature maps to capture the most salient features. After the convolutional
layers, a Time Distributed layer is applied to flatten the feature maps, preparing them for input into the Recurrent Neural Network
(RNN) layers. Two Long Short-Term Memory (LSTM) layers follow, with the first LSTM layer containing 128 units and
configured to return sequences.
![image](https://github.com/naseer30/ECG-CLASSIFICATION-USING-DEEP-LEARNING/assets/103625760/a0219d7a-742d-4a4e-a920-35bf0d2bb828)
IMPLEMENTATION:
The Dataset consists of 928 ECG images of Normal beat(284), Abnormal beat(233), Myocardial Infarction(239) and History of
MI(172), is divided into train, validation and test sets, then based on ratios ( 70% for training, 15% for validation, and 15% for
testing), the images are allocated to their respective sets. The number of images assigned to each set is proportional to the specified
ratios. We preprocessed the images before using the dataset of raw 12 Lead ECG report images to train the model. Before even
starting the preprocessing steps first we check the dimensions of the image, if they match with the dataset images dimensions it goes
to preprocessing or else if it does not match it will display a message saying that Uploaded image does not have suitable dimensions
for preprocessing
A. Data Pre-processing
1) Conversion to Grayscale: Firstly, the Raw ECG Images underwent a conversion process to Grayscale. This transformation
simplified data representation by eliminating color information and retaining only the intensity values, resulting in a singlechannel image representation.

![image](https://github.com/naseer30/ECG-CLASSIFICATION-USING-DEEP-LEARNING/assets/103625760/519135df-cf23-4dd4-ae6b-07b7161bd375)

![image](https://github.com/naseer30/ECG-CLASSIFICATION-USING-DEEP-LEARNING/assets/103625760/bbce068d-a294-4305-9ea1-1e18c941a39f)


3)	Division into Individual Leads: The Grayscale images were segmented into distinct leads to isolate specific electrical signals corresponding to different cardiac leads. This segmentation process relied on predefined coordinates delineating each lead's region of interest within the image, including top, bottom, left, and right boundaries, ensuring accurate extraction.

![image](https://github.com/naseer30/ECG-CLASSIFICATION-USING-DEEP-LEARNING/assets/103625760/0d40063c-8df1-4815-bb23-af28cae1a712)

![image](https://github.com/naseer30/ECG-CLASSIFICATION-USING-DEEP-LEARNING/assets/103625760/27473586-86ba-460c-b39e-6ade0961b495)
3) Gaussian Smoothing
To diminish noise and enhance uniformity, the lead images underwent Gaussian smoothing with a sigma value of 1. This procedure
helped blur edges, promoting a smoother appearance conducive to subsequent processing steps.
4) Thresholding
Global thresholding using Otsu’s method was applied to differentiate between foreground (signal) and background. By leveraging
an automated thresholding technique, optimal separation based on image intensity distribution was achieved, aiding in subsequent
segmentation.
5) Conversion to Binary Image
The thresholded image was converted into a binary format, where pixel values were assigned either 0 (background) or 255
(foreground). This conversion facilitated the segmentation of the ECG signal, streamlining subsequent analysis.
6) Conversion to Unsigned Byte Image
To ensure compatibility with subsequent processing steps and maintain pixel intensity information, the binary image was converted
to an unsigned byte image format, optimizing it for further analysis.
7) Standardization to Grayscale Image Mode
The image mode was standardized to 'L' (8-bit pixels, black and white), providing a consistent representation across all images and
facilitating seamless integration with subsequent analysis techniques.
8) Conversion to PIL Image Format
Finally, the preprocessed lead image was converted into a PIL (Python Imaging Library) image format. This conversion
ensured compatibility with various image manipulation libraries and enabled smooth integration with downstream tasks. These
preprocessing steps collectively aimed to enhance the clarity and quality of the ECG lead images, so that we can make predictions
based on all the Leads. These Preprocessed images are stored in separate folder and the 13 divided lead images are stored in
individual folder of each Image in four class folders for all three train, test and validation sets.
From all the above stages after dividing the leads ,the resulting preprocessed images are below

![image](https://github.com/naseer30/ECG-CLASSIFICATION-USING-DEEP-LEARNING/assets/103625760/1e816c3e-e2bf-440d-a760-d48e00ce12c0)

![image](https://github.com/naseer30/ECG-CLASSIFICATION-USING-DEEP-LEARNING/assets/103625760/731a05b3-d9c0-4c4e-819e-1f198facb260)
Now, for the training the model and validating, the preprocessed train and validation datasets directories are given. During training
in CRNN model, the model is compiled using the Adam optimizer with a learning rate of 0.001 and categorical cross-entropy loss
function. ModelCheckpoint and SaveHistory callbacks are utilized to save model weights and training history for each epoch,
respectively, while EarlyStopping helps prevent overfitting by monitoring validation loss. The model is trained for 20 epochs with
the specified callbacks and data generators for both training and validation sets.
B. Classification Process
Upon launching the application, users are prompted to upload an ECG image from their local system. Once an image is uploaded, it
is displayed in the user interface, providing users with a visual representation of the uploaded image. The application then proceeds
to preprocess the uploaded image, a crucial step in preparing the image for classification. During preprocessing, the uploaded image
undergoes several transformations, including conversion to grayscale and division into individual leads. These preprocessing steps
are essential for extracting relevant features from the ECG image and facilitating accurate classification. The grayscale version of
the image and its divided leads are displayed in expandable sections within the user interface, allowing users to inspect the
preprocessing results visually. After preprocessing, the application employs a pre-trained deep learning models (CNN, CRNN) to
classify the ECG image. The preprocessed leads are resized to a standard size and fed into the model for prediction. The model
predicts the class probabilities for each lead, and these predictions are displayed in the interface. The predicted class probabilities for
each lead are displayed, providing insights into the model's confidence in its predictions for individual leads. Additionally, the
overall predicted class for the ECG image is determined based on the aggregated predictions from all leads. This information is
presented to the user, enabling them to understand the classification outcome and make informed decisions based on the model's
predictions. All things considered, the application provides an interactive and intuitive platform for the classification of ECG images,
enabling users to efficiently examine and understand their ECG data.
RESULTS:
The model performed well on unseen data, and the classification results of unseen test data also shown good results. Both the CNN
and CRNN models demonstrated strong performance in training and validation stages, showcasing their ability to effectively
classify ECG images. Their precision, recall, and F1-score metrics indicate their reliability in accurately identifying patterns in heart
activity. These results highlight the models' potential in contributing significantly to medical diagnosis and advancing the field of
ECG image classification.
The CNN model demonstrated exceptional performance during training, achieving an accuracy of 97.82%. Even on unseen
validation data, it maintained accuracy of 93.40%. Moreover, its precision stands at 93.45%, with a recall of 93.38% and an overall
F1-score of 93.31%. Meanwhile, the proposed CRNN model exhibited a training accuracy of 98.82% and a validation accuracy of
93.72%. Notably, it achieved a precision of 93.74%, matching its recall rate of 93.72%, resulting in a robust F1-score of 93.70%.
These results underscore the effectiveness of both models in accurately classifying ECG images, promising significant contributions
to the field.

![image](https://github.com/naseer30/ECG-CLASSIFICATION-USING-DEEP-LEARNING/assets/103625760/27dcf340-a114-42d7-886b-36f65849c26c)
The above plots indicate the training and validation Accuracy and loss of CNN model. The below plots indicate the CRNN model’s train and validation accuracy and loss. 

![image](https://github.com/naseer30/ECG-CLASSIFICATION-USING-DEEP-LEARNING/assets/103625760/f4f5b6a9-cb06-47ad-b53e-23dd338861d9)
How to run the project:
I have uploaded all the code files steps - You should follow this order when you run the files download the dataset (ECG Images dataset - https://data.mendeley.com/datasets/gwbz3fsgp8/2)
1.run datasplit.py,
2.run Ecg.py, 
3.run cnn_model.py, 
4.run CRNN.py, 
5.run app.py (to run follow this cmd: streamlit run app.py),
6.run app2.py (to run follow this cmd: streamlit run app2.py), 
7.run evaluation.py (to run follow this cmd: streamlit run evaluation.py), 
8.run evaluation_2.py (to run follow this cmd: streamlit run evaluation_2.py).










