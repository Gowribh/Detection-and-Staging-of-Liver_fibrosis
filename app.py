from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image

app = Flask(__name__)
model = load_model('model/fibrosis_model.h5')

LABELS = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

# Precautions corresponding to each fibrosis stage
PRECAUTIONS = {
    "Stage 1": "Maintain a healthy diet, avoid alcohol, exercise regularly, and monitor liver function annually.",
    "Stage 2": "Adopt a liver-friendly diet, reduce weight if obese, avoid medications that stress the liver, and have regular checkups.",
    "Stage 3": "Consult a hepatologist, avoid all toxins, follow strict diet restrictions, and monitor progression every 3–6 months.",
    "Stage 4": "Strict medical supervision required. Focus on managing cirrhosis, consider transplantation options, and avoid all liver toxins."
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_path = "static/uploads/" + file.filename
        file.save(file_path)

        img = preprocess_image(file_path)
        prediction = model.predict(img)
        stage = LABELS[np.argmax(prediction)]
        precaution = PRECAUTIONS[stage]

        return render_template('index.html', filename=file.filename, stage=stage, precaution=precaution)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
