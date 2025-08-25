# Volatility Prediction using LSTM, Transformer, and Hybrid Models

This project explores different deep learning models for predicting stock market volatility. It utilizes Variational Mode Decomposition (VMD) to decompose the volatility signal into intrinsic mode functions (IMFs) and then uses LSTM, Transformer, and a Hybrid LSTM-TCN-Attention model to make predictions on these modes.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Project Overview

The goal of this project is to predict stock market volatility. Volatility is a key indicator in financial markets, and accurate predictions can be valuable for risk management and trading strategies. The project investigates the effectiveness of various neural network architectures, including:

- **LSTM (Long Short-Term Memory):** A type of recurrent neural network well-suited for sequential data like time series.
- **Transformer:** A model that uses self-attention mechanisms to weigh the importance of different parts of the input sequence.
- **Hybrid LSTM-TCN-Attention Model:** A custom model that combines LSTM, Temporal Convolutional Networks (TCN), and Self-Attention to leverage the strengths of each architecture.

The project also incorporates Variational Mode Decomposition (VMD) as a pre-processing step to decompose the volatility signal into different frequency components (IMFs), which are then used as inputs to the models.

## Data

The project uses historical stock market data, specifically focusing on the IT sector. The data includes columns such as 'Date', 'Open', 'High', 'Low', 'Close', 'Shares Traded', and 'Turnover (Cr)'.

The 'volatility' is calculated as the rolling standard deviation of the percentage change in the 'Close' price.

## Methodology

1. **Data Loading and Preprocessing:** Load the historical stock data, handle missing values, and calculate the 'returns' and 'volatility'.
2. **Data Scaling:** Scale the 'volatility' data using `StandardScaler` for optimal model performance.
3. **Data Preparation for Time Series:** Create sequences of data using a sliding window approach for training and testing the models.
4. **Variational Mode Decomposition (VMD):** Apply VMD to the volatility data to decompose it into intrinsic mode functions (IMFs). The optimal number of modes (K) is determined using AIC/BIC and reconstruction error methods.
5. **Model Building and Training:**
    - Build and train an LSTM model for volatility prediction.
    - Build and train a Transformer model for volatility prediction.
    - Build and train a Hybrid LSTM-TCN-Attention model, which takes the decomposed IMFs from VMD as input.
6. **Model Evaluation:** Evaluate the performance of each model using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
7. **Prediction and Visualization:** Generate predictions using the trained models and visualize the actual vs. predicted volatility.

## Models

- **LSTM Model:** A standard LSTM network with multiple layers and dropout for regularization.
- **Transformer Model:** A custom Keras model built with transformer encoder blocks.
- **Hybrid LSTM-TCN-Attention Model:** A PyTorch model that processes each IMF from VMD separately using parallel LSTM and TCN-Attention layers, and then combines their outputs for the final prediction.

## Results

The project compares the performance of the three models based on their evaluation metrics (MSE, MAE, RMSE). The visualizations show how well each model's predictions align with the actual volatility.

*(Note: The comparison in the notebook suggests the LSTM model performed better based on Test RMSE and Test Loss, but the Hybrid model is implemented using PyTorch and its performance is evaluated separately.)*

## Setup

1. **Clone the repository:** (If applicable, add instructions on how to get the code)
2. **Install dependencies:**(The notebook will install the necessary libraries)
3. **Test On Custom Dataset**(Change the data file with the custom data of yours, in the starting of both notebooks)


## Reference

```bibtex
@article{LIU2024121708,
title = {A stock series prediction model based on variational mode decomposition and dual-channel attention network},
journal = {Expert Systems with Applications},
volume = {238},
pages = {121708},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.121708},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423022108},
author = {Yepeng Liu and Siyuan Huang and Xiaoyi Tian and Fan Zhang and Feng Zhao and Caiming Zhang},
keywords = {Stock series prediction, Variational mode decomposition, Dual-channel attention network}
}
