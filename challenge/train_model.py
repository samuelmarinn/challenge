import pandas as pd
from challenge.model import DelayModel
import joblib

from challenge.settings import MODEL_NAME, TRAIN_DATA_PATH

train_df = pd.read_csv(TRAIN_DATA_PATH)

model = DelayModel()
X, Y = model.preprocess(train_df, target_column='delay')
model.fit(X, Y)

joblib.dump(model, f"models/{MODEL_NAME}")
print(f"Model saved as {MODEL_NAME}")
