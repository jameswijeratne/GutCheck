from flask import Flask, request, jsonify
import pandas as pd
import torch
from cnn import ConfidenceCNN, predict_confidence_score, scaler  # adjust imports

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConfidenceCNN()

@app.route("/predict", methods=["POST"])
def predict_from_csv():
    try:
        file = request.files["file"]
        df = pd.read_csv(file, error_bad_lines=False)

        # Select only the alpha and gamma columns
        df_filtered = df[[col for col in df.columns if 'alpha' in col.lower() or 'gamma' in col.lower()]]

        scores = predict_confidence_score(model=model, data=df_filtered, scaler=scaler, device=device)
        return jsonify({"confidence_scores": scores})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def test_model_on_local_file():
    try:
        df = pd.read_csv(r"C:/Users/James/Desktop/neurotech-hackathon-2025/train_data/testC.csv")
        df_filtered = df[[col for col in df.columns if 'alpha' in col.lower() or 'gamma' in col.lower()]]

        scores = predict_confidence_score(model=model, data=df_filtered, scaler=scaler, device=device)
        print(f"Confidence Scores for TEST:")
        for i, score in enumerate(scores):
            print(f"Window {i + 1}: {score:.4f}")
    except Exception as e:
        print("Error during testing:", e)

if __name__ == "__main__":
    test_model_on_local_file()
    #app.run(debug=True, port=5000)
