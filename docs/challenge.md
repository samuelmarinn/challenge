## General cosiderations

- This proejct was developed using python 3.12.3
- Docker should be installed to make changes in the project
- For deployment in windows, is highly recommended to install WSL in order to run makefile locally
- venv is used as the vrtual environment package

## Model Development

### Considerations

- the packages versions are changed, as an specific python version wasn't provided
- in tests/model/test_model.py the data file path is modified, as it gave errors
- in challenge/settings.py, the cosntant TOP_10:FTS is defined according to the DS results

### Model selection

- According to he DS work, the best results are achieved with class balancing and feature selection. With those considerations during raining, between XGBoost and logistic regression model the difference in f1-score is minimum, which means that both models offer the same performance, so, in this case, the decision depends on other factors, such as simplicity, interpretability or flexibility. As the final prodcut of this work is an API, ***logistic regression*** is the model selected, as it offers more interpretability, wich might be important in the future of the project in the analysis of the obtained results, leving plenty of room for improvement

## Api development

### Condierations

- In tests/api/api_tests.py, the response code in all test_should_failed_unkown_column is changed from 400 to 422, as 422 is the correct code indicating incorrect values on columns
- In tests/api/api_tests.py, a default model and logistic regression are created, according to the implementation of the api

