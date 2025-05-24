from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
TEST_SIZE = 0.33

FTS_COLNAMES = ['OPERA', 'MES', 'TIPOVUELO']
TOP_10_FTS = [
    "OPERA_Latin American Wings", 
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
] ### according to DS on explotarion.ipynb


BEST_MODEL = LogisticRegression() ### check data/challenge.md to see model selection
TRAIN_DATA_PATH = "data/data.csv"
MODEL_NAME = 'delay_model_logreg.pkl'
