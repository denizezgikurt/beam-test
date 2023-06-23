def load_models():
    # This runs exactly once when the app first starts
    model = pipeline(
        "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
    )

    return model

def run_inference(**inputs):
    # Retrieve the model from the loader
    model = inputs["context"]

    result = model(inputs["text"], truncation=True, top_k=2)
    prediction = {i["label"]: i["score"] for i in result}

    return {"prediction": prediction}
