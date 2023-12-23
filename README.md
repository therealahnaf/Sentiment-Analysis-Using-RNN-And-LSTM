Introduction:

This report investigates sentiment analysis using IMDb movie reviews as a dataset, employing three distinct models: a Shallow RNN, a Unidirectional LSTM model, and a Bidirectional LSTM model. The goal is to assess the ability of these recurrent neural network architectures in capturing sentiments within the reviews. By training and evaluating these models, we explore their comparative strengths and limitations in sentiment analysis tasks. The findings of this study show how each model differs from one another in terms of accuracy and time complexity.

Dataset Description:

Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Our dataset consists of the text in the review and the sentiment label for each review.
review: This is the sequence of sentences for each review.
sentiment: This is the sentiment label for each review.

Data-Preprocessing:

Splitting data:

At first, we extract the data from the csv file and split it into testing and training sets with 20% and 80% of the data respectively. The random_state variable is set to 1000 which means we randomly pick the data from the dataset while splitting to ensure there is no connection in the order of the data.
 
Tokenizing text sequences to word sequences:

The review data is initially a string of words. In order to train the neural network models, we need to convert them to integers which is done with the use of Tokenizer from keras. Each unique word from the sentences is given a unique integer value and a list of those sequences of integers is returned upon tokenizing. The vocabulary size is then assigned to the vocab_size variable using the length of the list containing all the unique words in the dataset. This list is found from the tokenizer itself where the index 0 is reserved for other purposes, thus, we add 1 to get the actual count.

Adding Padding and Encoding Target Values:

The encoding of the reviews section of the dataset is done through tokenizing, however, the target, sentiment, values need to be encoded as well. We use LabelEncoder from sklearn which encodes “positive” and “negative” values as 0s and 1s respectively. In order to ensure that all inputs are of uniform length, we introduce padding with maxlen set to 256 and padding set to “post”. This adds a sequence of 0s at the end of each review until the length of the input is 256.

Model Descriptions:

Each of our models uses an embedding layer and a dense output layer. For all three models, we use RELU and sigmoid as the activation function for the second and third layer respectively. The three models used for the prediction are - Shallow RNN model, Bi-directional LSTM model, and Uni-directional LSTM model.
Shallow RNN model: In this model, between the embedding layer and the dense output layer, another dense layer is added with an output shape of 10.
Unidirectional LSTM model: In this model, we replace the first dense layer with a unidirectional LSTM layer of output shape 10.
Bidirectional LSTM model: For our third model, we replace the unidirectional LSTM layer with a bidirectional LSTM layer.
Training And Testing
The models are trained for 20 epochs with a batch size of 10, which means weights are updated after every 10 input. 

Result Analysis:

From the results obtained, we can infer:
The Shallow RNN model struggles to capture patterns compared to the other models, indicating the need for a deeper network or more complex layers such as LSTM.
The Unidirectional LSTM model's high training accuracy suggests overfitting, emphasizing the importance of regularization techniques to improve generalization. This model achieves 100% training accuracy after around 8 epochs.
The Bidirectional LSTM model, while still exhibiting overfitting, demonstrates better testing accuracy, showcasing the effectiveness of bidirectional information flow in understanding sentiment context. In contrast to the unidirectional model, this model achieves 100% training accuracy after around 3 epochs.

We have achieved the best accuracy with the bidirectional LSTM model. The bidirectional architecture allows the model to capture contextual information from both past and future tokens, enabling a more comprehensive understanding of the sequential nature of sentiment in the IMDb reviews dataset. Some of the recommendations to improve these models are:
Experiment with regularization techniques (e.g., normalization) to address the overfitting in the LSTM models.
The overfitting problem could also be addressed by taking a smaller percentage of the dataset as training data (eg. 65% - 70% instead of 80%).
Instead of using unidirectional LSTM and bidirectional LSTM, we could use unidirectional GRU and bidirectional GRU for the second and third models respectively to increase the speed of the training process. 

