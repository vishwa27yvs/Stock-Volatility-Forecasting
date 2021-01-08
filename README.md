# Stock-Volatility-Forecasting

Earning calls have a substantial impact on the volatility in the stock price and can be used to analyse a stock’s risk level. Along with textual features, the vocal features and voice tones while delivering the earning call also are depictive of the firm’s performance. This approach uses an audio encoder and text encoder for obtaining the embeddings and features, this combined encoded representation has been used along with past volatility values to predict the volatility of the stock over the next t=3 days. The model has been built on the earning calls data available for S&P 1500 companies.

# data_feature_extn.py: 
Contains code for
1) Scraping stock data
2) Computing sentence embeddings from SentenceBERT
3) Aligning audio features
4) Padding audio and text feature input
5) Computing volatility from closing prices

# models.py : Contains the pipeline for the prediction. 
1)Text encoder
2) Audio Encoder 
3) Fusing and combining with past volatility values
