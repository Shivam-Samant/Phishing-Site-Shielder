from fastapi import FastAPI
from pydantic import BaseModel
from temp_helper import load_model, getFeatures
import pandas as pd
app = FastAPI()

class Sites(BaseModel):
    url: str

@app.post("/predict")
def verifyUrlLegitimacy(request: Sites):
    features = getFeatures(request.url)
    df = pd.DataFrame(features, index=[0])
    model = load_model('phishingSiteShielderNB')
    ans = model.predict(df)
    # good = 1 & bad = 0
    return ans.tolist()
