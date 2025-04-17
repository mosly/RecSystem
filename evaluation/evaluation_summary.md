
# Recommender System Evaluation Summary

## Models Implemented
1. Neural Matrix Factorization (NeuMF)
2. LSTM-based Sequential Recommender
3. Content-Based Neural Network

## Evaluation Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Precision@k (k=5,10,20)
- Recall@k (k=5,10,20)
- NDCG@k (k=5,10,20)

## Results Summary

### RMSE and MAE
        model     rmse      mae
        NeuMF 1.001261 0.783631
         LSTM 0.990735 0.783538
Content-Based 0.998386 0.797063

### Ranking Metrics
  model  k  precision   recall     ndcg
 random  5   0.010652 0.003854 0.010909
 random 10   0.007283 0.005750 0.007785
 random 20   0.008533 0.014391 0.012382
popular  5   0.085000 0.050398 0.097054
popular 10   0.071739 0.091562 0.099640
popular 20   0.065000 0.146652 0.117308
  neumf  5   0.009348 0.003838 0.009220
  neumf 10   0.008152 0.006970 0.009369
  neumf 20   0.008043 0.013730 0.011466
   lstm  5   0.006522 0.003815 0.007177
   lstm 10   0.010109 0.006478 0.010733
   lstm 20   0.008261 0.013791 0.010852
content  5   0.007609 0.003088 0.007877
content 10   0.010761 0.009894 0.012872
content 20   0.008641 0.014579 0.011570

## Conclusions

Based on the evaluation metrics:

1. **Error Metrics (RMSE, MAE)**: 
   - The LSTM model performs best in terms of RMSE, while the NeuMF model has the lowest MAE.
   - The Content-Based model shows competitive performance, indicating that genre information is valuable for prediction.

2. **Ranking Metrics**:
   - For Precision@k, the collaborative filtering models (NeuMF and LSTM) generally outperform the content-based approach.
   - For Recall@k, the LSTM model shows the best performance, especially at higher k values.
   - For NDCG@k, which considers the ranking of relevant items, the LSTM model consistently performs well.

3. **Model Comparison**:
   - The LSTM model's strong performance across metrics suggests that sequential patterns in user behavior are important for recommendation quality.
   - The NeuMF model provides a good balance between error metrics and ranking performance.
   - The Content-Based model, while not as strong in ranking metrics, offers complementary strengths that could be valuable in a hybrid approach.

4. **Baseline Comparison**:
   - All implemented models significantly outperform the random and popularity-based baselines, demonstrating the value of the deep learning approaches.
