import beam
from transformers import pipeline


app = beam.App(
    name="sentiment-analysis-app",
    cpu=4,
    memory="16Gi",
    python_version="python3.8",
    python_packages=["transformers", "torch", "numpy"],
)



def predict_sentiment(**inputs):
    model = pipeline(
        "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
    )
    result = model(inputs["text"], truncation=True, top_k=1)
    prediction = {i["label"]: i["score"] for i in result}

    print(prediction)

    return {"prediction": prediction}
