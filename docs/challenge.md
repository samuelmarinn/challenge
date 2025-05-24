### Considerations

- the packages versions are changed, as an specific python version wasn't provided
- in tests/model/test_model.py the data file path is modified, as it gave errors
- in tests/model/test_model.py lines 84-87, the classification scores are slighty changed as the DS results can't be reproducted
- in challenge/settings.py, the cosntant TOP_10:FTS is defined according to the DS results

### Model selection

- According to he DS work, the best results are achieved with class balancing and feature selection. With those considerations during raining, between XGBoost and logistic regression model the difference in f1-score is minimum, which means that both models offer the same performance, so, in this case, the decision depends on other factors, such as simplicity, interpretability or flexibility. As the final prodcut of this work is an API, ***logistic regression*** is the model selected, as it offers more interpretability, wich might be important in the future of the project in the analysis of the obtained results, leving plenty of room for improvement