# Machine Learning Project
# Modeling Stock Market Returns using Statistical, Classical Machine Learning and Deep Learning Methods

# Dataset

[CNNPred Dataset](https://archive.ics.uci.edu/dataset/554/cnnpred+cnn+based+stock+market+prediction+using+a+diverse+set+of+variables)
 This dataset contains several daily features of S&P 500, NASDAQ Composite, Dow Jones Industrial Average, RUSSELL 2000, and NYSE Composite from 2010 to 2017.


# Baselines

- ## ARIMA
  Widely used for prediction in time series data modeling, ARMA requires the time series to be stationary; which means a constant mean, constant variance and non-seasonal. Since, stock market prices don’t have constant mean, a tranformed feature **y<sub>i</sub> = price<sub>i</sub> − price<sub>i−d</sub>** where **d** is the lag. Auto-Regressive(AR) and
Moving Average(MA) models are parameterized by auto-correlation coefficient(**p**) and Partial auto-correlation coefficient(**q**). Search is performed given the maximum values of **p** and **q** to find the best model.
     - ### Results
          - An RMSE of 423.24 was observed.However, if the first testing point is included in the training data to predict the next point, an RMSE of 49.72 was observed. 



- ## PCA + DNN
  This model has been given in `https://jfin-swufe.springeropen.com/articles/10.1186/s40854-019-0138-0` for forecasting daily return direction of the
SPDR S&P 500 ETF index. All of the 60 financial variables evaluated in the study’s dataset are already present in & analogous to the 82 features of the chosen dataset, CNNPred. Zhong and Enke explored multiple data transformation techniques including PCA and its variants, fuzzy robust principal component analysis (FR-
PCA) and kernel-based principal component analysis (KPCA), among others. Their results showed that traditional PCA outperformed all non-linear techniques on real-world data. Thus, PCA is chosen as the data transformation technique and PCA represented dataset with 82 principal components is used.
    - ### Data Preprocessing
        - A classical statistical principle is used for detection of outliers based on inter-quartile ranges. These outliers are accordingly adjusted similar to a method used in `https://link.springer.com/article/10.1007 s005210170010`. The cleaned data is split in 70/15/15 ratio for train, validation and test dataset respectively. Data is standardized with the mean and variance of the training dataset.

    - ### Model & Training
        - PCA-represented dataset is classified using neural network comprising of 4 layers with RELU activation & Sigmoid activation for the last layer. Dropout has been introduced in the network to avoid overfitting. Binary cross entropy loss is used as the loss criterion. Initial learning rate set to 0.0001 with ADAM optimizer for training over maximum of 100 epochs.

    - ### Results
        - Obtained an accuracy of 0.559 and a F1-score of 0.656 on the test set.



- ## CNN-Pred2D 
  This model given in `https://www.sciencedirect.com/science/article/abs/pii/S0957417419301915` classifies the change in the closing price of the market using only the data for the market under analysis. It takes the input of all the features from last 60 days and leverages 2D convolution filters for making feature maps and finally classifying the change in price. Since CNNs are good at capturing short range data and hierarchically extract features from day wise data, they serve as a good method for predicting stock prices. The initial learning rate for training this network was set to 0.001, and the ADAM optimizer was used for training over 100 epochs. Since the network was quite small with only one fully connected layer and three convolution layers, a weight decay of 0.0001 was sufficient to regularize the network. Each feature was normalized according to the training set and was used during the validation and test time.

    - ### Results
        - F-Scores on the test set using CNN-Pred2D trained for 100 Epochs on the CNNPred Stock market Dataset are given below:
       
          | Market Name | Average | Maximum |
          | ----------- | ------- | ------- |
          | `NASDAQ`      | `0.438`   | `0.552`   |
          | `NYSE`        | `0.491`   | `0.492`   |
          | `S&P 500`     | `0.434`   | `0.573`   | 
          | `Russell`     | `0.497`   | `0.691`   |
          | `DJI`         | `0.471`   | `0.696`   |
             



- ## CNN-Pred3D
  This model classifies the change in the closing price of the market using the data of various markets. It takes the input as a 3D block of all the features from the 5
markets in the dataset over the last 60 days and leverages 3D convolution filters for making feature maps and finally classifying the change in price by using data across many markets. The usage of 3D convolution filters is the same as 2D convolution filters only that these filters hierarchically extract features and are temporally sensitive over an additional dimension. Similar to the CNN-Pred2D, the
initial learning rate for training this network was set to 0.001, and the ADAM optimizer was used for training over 100 epochs with a weight decay of 0.0001. Each feature was normalized according to the training set and was used during the validation and test time.
   - ### Results
        - F-Scores on the test set using CNN-Pred3D trained for 100 Epochs on the CNNPred Stock market Dataset are given below: 

          | Market Name | Average | Maximum |
          | ----------- | ------- | ------- |
          | `NASDAQ`      | `0.49`   | `0.536`   |
          | `NYSE`        | `0.432`   | `0.566`   |
          | `S&P 500`     | `0.491`   | `0.67`   |
          | `Russell`     | `0.489`   | `0.521`   |
          | `DJI`         | `0.486`   | `0.5`   |






- ## CNN-LSTM
  This model given in `https://www.hindawi.com/journals/complexity/2020/6622927` considers learning sequence aware features as the stock market is an
event which moves in the temporal dimension, it is difficult to ignore the sequential information present in the latent embeddings for the downstream tasks.
This work explored the price prediction perspective for more informed stock market trading and hence was a regression task. The convolutional features were used to
calculate the temporal features over time stamps and then an LSTM was used to capture the sequential features. The structure of LSTM is designed in such a manner
that it works on selectively learning which information to hide and which to infer on over a certain time step and pass both the states for the next time
step. This overparameterization leads to a delayed stability of LSTMs in terms of metrics but provably results to a more optimal result in lesser number of iterations. This model gave sufficiently good results on predicting the closing price of a market on training with 100 epochs with a learning rate of 0.001 on ADAM optimizer, with weight decay of 0.0001.

    - ### Results
        - Regression metrics on the test set using CNN-LSTM model trained for 100 Epochs on the CNNPred Stock market Dataset are given below:

          | Market Name | MAE | RMSE | R<sup>2</sup> | Max Closing Price |
          | ----------- | ------- | ------- | ------ |  ---------------  |
          | `NASDAQ`      | `645.21`   | `854.89`  | `0.99` | `341682.12` |  
          | `NYSE`        | `433.31`   | `577.70`   | `0.99` | `388508.06` | 
          | `S&P 500`     | `143.58`   | `192.79`   | `0.99` | `24420.32` |
          | `Russell`     | `162.90`   | `205.95`   | `0.99` | `12414.37` | 
          | `DJI`         | `1899.07`   | `2551.76`   | `0.99` | `2980330.8` |




# Flagship Model


- ## Neural ODE Classifiers
  This model uses the same inputs as the CNNPred Models with the same feature normalization performed. Used implementation of the forward and backward
pass for ODE solving and made a custom feature map extraction architecture with the ODE solving as a module taking the feature map as the input. The feature map extractor is a simple 2D CNN architecture which takes in the 3D input, as represented by an instance. Achieved faster training converging to a stable F-score in 15 epochs as compared to the 100 epochs of CNNPred2D and CNNPred3D. Got improved classification results on training this classifier on both CNNPred Dataset and CNNPred3D Dataset. The optimizer used was ADAM with a learning rate of 0.001 for 15 epochs.
     
   - ### Results
        - F-Scores on the test set using infinite depth classifier using NeuralODE trained on the CNNPred Stock market Dataset on single market data as input are gien below:

          | Market Name | Average | Maximum |
          | ----------- | ------- | ------- |
          | `NASDAQ`      | `0.48`   | `0.579`   |
          | `NYSE`        | `0.531`   | `0.567`   |
          | `S&P 500`     | `0.542`   | `0.622`   |
          | `Russell`     | `0.517`   | `0.584`   |
          | `DJI`         | `0.519`   | `0.589`   |

        - F-Scores on the test set using infinite depth classifier using Neural ODE trained for 15 Epochs on the CNNPred3D Stock market Dataset on all the market data as input are given below:
          
          | Market Name | Average | Maximum |
          | ----------- | ------- | ------- |
          | `NASDAQ`      | `0.438`   | `0.55`   |
          | `NYSE`        | `0.567`   | `0.624`   |
          | `S&P 500`     | `0.541`   | `0.556`   |
          | `Russell`     | `0.491`   | `0.495`   |
          | `DJI`         | `0.521`   | `0.527`   |





- ## Neural-ODE VAE 
  Used GRU encoder to encode the sequence to a hidden state and Neural ODE as a decoder. The encoder-decoder networks were trained on a data of 60 days and during inference time, data of 59 consecutive days were given in order for the model to predict the closing price for the 60th day while ODE solving the ODE decoder using Euler’s method. Only used two dimensional data consisting of the generated time-steps and the closing price on which we have done the training and made the inferences. Used a latent dimension of 4 for the GRU and trained for 100 epochs using the ADAM optimizer with learning rate of 0.001. Although inferior results were achieved as compared to CNN-LSTM, concluded from this set of experiments that the data was not sufficient in training the ODE-VAE model as generative model such as VAEs require a lot of training samples. Furthermore, also made the ODE on the closing price as that was the only available data in the dataset. Prior Knowledge of the exact opening, closing, and median prices would have been a more ideal setting in training this model.
     
   - ### Results
        - Regression metrics on the test set using ODE VAE model trained for 150 Epochs on the sequential closing price data of the CNNPred Stock market Dataset are given below: 

          | Market Name | MAE | RMSE | Max Closing Price |
          | ----------- | ------- | ------- | -------|
          | `NASDAQ`      | `2177.418`   | `2251.247`  | `341682.12` |  
          | `NYSE`        | `2461.052`   | `2538.778`   | `388508.06` | 
          | `S&P 500`     | `799.763`   | `815.964`   | `24420.32` |
          | `Russell`     | `1350.78`   | `1355.90`   | `12414.37` | 
          | `DJI`         | `7288.813`   | `7455.721`   | `2980330.8` |



- ## GAN
  The GAN model builds upon the idea to predict PCA transformed features as utilized previously and is tested on the S&P 500 index. Initially data is preprocessed with outlier adjustments as similar to and then the data is split in 80/20 ratio for train and test set respectively. Also, data is standardized with the mean and variance of the training dataset. The generator utilizes a shallow LSTM single layer network followed by a fully connected layers comprising of Tanh and ReLU activations along with batch normalization. The generator’s aim is to take features of past 60 days and generate the new features of the 61st day. The discriminator is a deep CNN network that takes in features of 61 days and predicts whether the this trend is real or fake. The loss function used to train the discriminator is BCE Loss. Finally, used the output of the generator to predict the direction of the closing price of the index.
     
   - ### Results
        - The accuracy on the train set and the test set were observed to be 0.545 & 0.504 respectively.
