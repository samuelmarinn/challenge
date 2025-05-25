## General considerations

- This project was developed using python 3.12.3, as an specific python version wasn't provided
- Docker should be installed to test locally this project
- For development in windows, is highly recommended to install WSL in order to run the makefile locally
- venv is used as the virtual environment package
- the packages versions in the requirement files are changed according to the used python version, 

## Model Development

### Considerations

- in tests/model/test_model.py the data file path is modified, as it gave errors
- in challenge/settings.py, the constant TOP_10_FTS is defined according to the DS results presented in the exploration notebook
- a train_model.py file is added, used locally to create the model used in the api

### Model selection

- According to the DS work, the best results are achieved with class balancing and feature selection. With those considerations during raining, between XGBoost and logistic regression model the difference in f1-score is practically zero, which means that both models offer the same performance, so, in this case, the decision depends on other factors, such as simplicity, interpretability or flexibility. As the final product of this work is an API, ***logistic regression*** is the model selected, as it offers more interpretability, which might be important in the future of the project in the analysis of the results obtained. It is highly encouraged to run again the exploration notebook if the training dataset changes, as it might change the model performance, and thus, changing the decision made in this version.

## Api development

### Considerations

- In tests/api/api_tests.py, the response code in all test_should_failed_unkown_column is changed from 400 to 422, as 422 is the correct code indicating incorrect values on columns
- In tests/api/api_tests.py, a default model and logistic regression are created, according to the implementation of the api and model


