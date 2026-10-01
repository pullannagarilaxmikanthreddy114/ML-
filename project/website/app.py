import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Ensure compatibility when unpickling models trained on scikit-learn 1.6 in newer scikit-learn
try:
    import sklearn.compose._column_transformer as ct
    if not hasattr(ct, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

# Paths - check both website folder and parent workspace folder
SITE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SITE_DIR)

def resolve_file(rel_paths):
    for base in [SITE_DIR, PARENT_DIR]:
        for rel in rel_paths:
            candidate = os.path.join(base, rel)
            if os.path.isfile(candidate):
                return candidate
    return os.path.join(SITE_DIR, rel_paths[0])

def resolve_dir(rel_paths):
    for base in [SITE_DIR, PARENT_DIR]:
        for rel in rel_paths:
            candidate = os.path.join(base, rel)
            if os.path.isdir(candidate):
                return candidate
    return os.path.join(SITE_DIR, rel_paths[0])

MODEL_PATH = resolve_file([
    os.path.join('model', 'best_parking_prediction_model.pkl'),
    os.path.join('model-ML', 'best_parking_prediction_model.pkl')
])
DATASET_PATH = resolve_file([
    os.path.join('dataset', 'final_clean_parking_dataset.csv')
])
METRICS_PATH = resolve_dir(['metrics'])
RESULTS_PATH = resolve_dir(['results'])
REPORTS_PATH = resolve_dir(['reports'])

# Load Model
print(f"Loading ML model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model unavailable; using fallback predictor: {e}")
    model = None

# Pre-load dataset metadata
locations_info = {}
analytics_data = {}

if os.path.exists(DATASET_PATH):
    try:
        df_dataset = pd.read_csv(DATASET_PATH)
        grouped = df_dataset.groupby('SystemCodeNumber')
        for loc, group in grouped:
            max_cap = int(group['Capacity'].max())
            avg_prev = float(group['Previous_Availability'].mean())
            avg_lag2 = float(group['Lag_2_Availability'].mean())
            avg_lag3 = float(group['Lag_3_Availability'].mean())
            avg_roll = float(group['Rolling_Average_Availability'].mean())
            locations_info[str(loc)] = {
                'capacity': max_cap,
                'avg_prev': round(avg_prev),
                'avg_lag2': round(avg_lag2),
                'avg_lag3': round(avg_lag3),
                'avg_roll': round(avg_roll)
            }

        hourly_trend = df_dataset.groupby('Hour')['OccupancyRate'].mean().round(3).to_dict()
        top_busy = df_dataset.groupby('SystemCodeNumber')['OccupancyRate'].mean().sort_values(ascending=False).head(8).round(3).to_dict()
        counts, bin_edges = np.histogram(df_dataset['OccupancyRate'].dropna() * 100, bins=5)
        dist_bins = [f"{int(bin_edges[i])}% - {int(bin_edges[i+1])}%" for i in range(len(counts))]

        analytics_data = {
            'hourly_hours': [f"{h}:00" for h in hourly_trend.keys()],
            'hourly_values': [round(v * 100, 1) for v in hourly_trend.values()],
            'busy_locations': list(top_busy.keys()),
            'busy_values': [round(v * 100, 1) for v in top_busy.values()],
            'dist_labels': dist_bins,
            'dist_counts': counts.tolist()
        }
    except Exception as e:
        print(f"Error processing dataset for analytics: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/locations', methods=['GET'])
def get_locations():
    return jsonify({
        'status': 'success',
        'locations': locations_info
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        system_code = str(data.get('SystemCodeNumber', 'BHMNCPNST01'))
        capacity = float(data.get('Capacity', 485))
        hour = int(data.get('Hour', 12))
        day = int(data.get('Day', 15))
        month = int(data.get('Month', 10))
        day_of_week = str(data.get('DayOfWeek', 'Monday'))
        is_weekend = int(data.get('IsWeekend', 0))
        
        prev_avail = float(data.get('Previous_Availability', 150))
        lag2_avail = float(data.get('Lag_2_Availability', 150))
        lag3_avail = float(data.get('Lag_3_Availability', 150))
        roll_avail = float(data.get('Rolling_Average_Availability', 150))

        input_data = pd.DataFrame([{
            'SystemCodeNumber': system_code,
            'Capacity': capacity,
            'Hour': hour,
            'Day': day,
            'Month': month,
            'DayOfWeek': day_of_week,
            'IsWeekend': is_weekend,
            'Previous_Availability': prev_avail,
            'Lag_2_Availability': lag2_avail,
            'Lag_3_Availability': lag3_avail,
            'Rolling_Average_Availability': roll_avail
        }])

        if model is not None:
            raw_pred = model.predict(input_data)[0]
        else:
            raw_pred = (
                0.40 * prev_avail
                + 0.25 * lag2_avail
                + 0.15 * lag3_avail
                + 0.20 * roll_avail
            )
        
        pred_avail = max(0.0, min(capacity, float(raw_pred)))
        pred_occupied = max(0.0, capacity - pred_avail)
        
        occ_rate = round((pred_occupied / capacity) * 100, 1) if capacity > 0 else 0.0
        avail_rate = round((pred_avail / capacity) * 100, 1) if capacity > 0 else 0.0

        if occ_rate >= 85.0:
            congestion_level = "High"
            congestion_desc = "High Demand - Parking almost full"
            badge_color = "#ef4444"
        elif occ_rate >= 55.0:
            congestion_level = "Moderate"
            congestion_desc = "Moderate Occupancy - Spaces filling up"
            badge_color = "#eab308"
        else:
            congestion_level = "Low"
            congestion_desc = "Ample Parking - Plenty of empty bays"
            badge_color = "#22c55e"

        return jsonify({
            'status': 'success',
            'prediction': {
                'raw_value': round(float(raw_pred), 2),
                'predicted_available_spaces': round(pred_avail, 1),
                'predicted_occupied_spaces': round(pred_occupied, 1),
                'capacity': capacity,
                'occupancy_rate_pct': occ_rate,
                'availability_rate_pct': avail_rate,
                'congestion_level': congestion_level,
                'congestion_desc': congestion_desc,
                'badge_color': badge_color
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    try:
        comp_csv = os.path.join(METRICS_PATH, 'model_comparison.csv')
        feat_csv = os.path.join(RESULTS_PATH, 'final_feature_importance.csv')
        pred_csv = os.path.join(RESULTS_PATH, 'final_model_predictions.csv')

        comparison = []
        feature_imp = []
        actual_vs_pred = {}

        if os.path.exists(comp_csv):
            comparison = pd.read_csv(comp_csv).to_dict(orient='records')

        if os.path.exists(feat_csv):
            feature_imp = pd.read_csv(feat_csv).head(8).to_dict(orient='records')

        if os.path.exists(pred_csv):
            df_p = pd.read_csv(pred_csv).head(50)
            actual_vs_pred = {
                'labels': [f"Point {i+1}" for i in range(len(df_p))],
                'actual': df_p['Actual_AvailableSpaces'].tolist(),
                'predicted': df_p['Predicted_AvailableSpaces'].round(1).tolist()
            }

        return jsonify({
            'status': 'success',
            'model_comparison': comparison,
            'feature_importance': feature_imp,
            'actual_vs_pred': actual_vs_pred
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    return jsonify({
        'status': 'success',
        'analytics': analytics_data
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on http://localhost:{port} (http://127.0.0.1:{port})")
    app.run(host='0.0.0.0', port=port, debug=False)
