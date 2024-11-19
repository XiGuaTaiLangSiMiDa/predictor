import os
from flask import render_template, request, jsonify, send_file
from app import app
from app.models.predictor import CryptoPredictor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    token = request.form.get('token', 'GMT')
    pair = f"{token}/USDT"
    
    predictor = CryptoPredictor()
    predictor.symbol = pair
    
    try:
        predictions, historical_data = predictor.generate_predictions()
        if predictions is not None:
            plot_path = os.path.join(app.root_path, 'static', 'plots', f'{token.lower()}_prediction.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            predictor.plot_predictions(predictions, historical_data, plot_path)
            
            prediction_data = []
            for date, row in predictions.iterrows():
                prediction_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'price': f"{row['ensemble']:.4f}"
                })
            
            return jsonify({
                'success': True,
                'predictions': prediction_data,
                'plot_url': f'/static/plots/{token.lower()}_prediction.png'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': str(datetime.now())
    })

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500