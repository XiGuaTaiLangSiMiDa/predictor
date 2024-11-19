from flask import Flask, request, render_template, send_file
from crypto_predictor import CryptoPredictor
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    token = request.form.get('token', 'GMT')
    pair = f"{token}/USDT"
    
    predictor = CryptoPredictor()
    predictor.symbol = pair
    
    historical_data = predictor.fetch_historical_data()
    if historical_data is not None:
        predictions = predictor.generate_predictions()
        if predictions is not None:
            predictor.plot_predictions(predictions, historical_data)
            
            prediction_data = []
            for date, row in predictions.iterrows():
                prediction_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'price': f"{row['ensemble']:.4f}"
                })
            
            return {
                'success': True,
                'predictions': prediction_data,
                'plot_updated': True
            }
    
    return {
        'success': False,
        'error': 'Failed to generate predictions'
    }

@app.route('/plot')
def get_plot():
    return send_file('prediction_plot.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=3333)
