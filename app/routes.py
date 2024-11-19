import os
from flask import render_template, request, jsonify, send_file
from app import app
from app.models.predictor import CryptoPredictor
from datetime import datetime

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
        # First fetch historical data
        historical_data = predictor.fetch_historical_data()
        if historical_data is None:
            app.logger.error("Failed to fetch historical data")
            return jsonify({
                'success': False,
                'error': 'Failed to fetch historical data from exchange'
            })

        if historical_data.empty:
            app.logger.error("Historical data is empty")
            return jsonify({
                'success': False,
                'error': 'No historical data available for this token'
            })

        # Generate predictions
        try:
            predictions, historical_data = predictor.generate_predictions()
        except Exception as pred_error:
            app.logger.error(f"Prediction generation error: {str(pred_error)}")
            return jsonify({
                'success': False,
                'error': f'Failed to generate predictions: {str(pred_error)}'
            })

        if predictions is None or predictions.empty:
            app.logger.error("No predictions generated")
            return jsonify({
                'success': False,
                'error': 'Failed to generate predictions'
            })

        # Create plots directory if it doesn't exist
        plot_filename = f'{token.lower()}_prediction.png'
        plot_path = os.path.join(app.root_path, 'static', 'plots', plot_filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        
        # Generate plot
        try:
            predictor.plot_predictions(predictions, historical_data, plot_path)
            if not os.path.exists(plot_path):
                raise Exception("Plot file was not created")
        except Exception as plot_error:
            app.logger.error(f"Plot error: {str(plot_error)}")
            return jsonify({
                'success': False,
                'error': f'Failed to generate plot: {str(plot_error)}'
            })

        # Prepare prediction data
        try:
            prediction_data = []
            for date, row in predictions.iterrows():
                prediction_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'price': f"{row['ensemble']:.4f}"
                })
            
            if not prediction_data:
                raise Exception("No prediction data generated")

            # Return response with plot_url matching the HTML expectation
            response_data = {
                'success': True,
                'predictions': prediction_data,
                'plot_url': f'/static/plots/{plot_filename}'
            }
            app.logger.info(f"Successfully generated predictions for {token}")
            return jsonify(response_data)
        except Exception as format_error:
            app.logger.error(f"Data formatting error: {str(format_error)}")
            return jsonify({
                'success': False,
                'error': f'Failed to format prediction data: {str(format_error)}'
            })
    except Exception as e:
        app.logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
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
