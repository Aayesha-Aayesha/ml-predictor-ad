from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
#modeltrainon()

scaler = joblib.load('./scaler.pkl')
model = joblib.load('./random_forest_model.pkl')

# Route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_data():
    param1 = request.json.get('param1')
    param2 = request.json.get('param2')
    param3 = request.json.get('param3')
    param4 = request.json.get('param4')
    features = np.array([[param1, param2, param3, param4]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    #return jsonify({'deposit_next': prediction[0]})
    return jsonify(result= prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)