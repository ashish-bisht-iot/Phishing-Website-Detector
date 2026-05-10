import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from feature_extractor import extract_features, features_to_vector, FEATURE_VECTOR_KEYS

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'phishing_model.pkl')

_model_cache = None


def load_model():
    """Load model from disk (cached after first call)."""
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python train_model.py` first."
            )
        with open(MODEL_PATH, 'rb') as f:
            _model_cache = pickle.load(f)
    return _model_cache


def get_risk_level(score: float) -> str:
    if score < 0.30:
        return 'safe'
    elif score < 0.65:
        return 'suspicious'
    else:
        return 'phishing'


def build_feature_summary(features: dict) -> list[dict]:
    """Return human-readable feature breakdown for the frontend."""
    summary = []

    def add(name, value, status, detail=''):
        summary.append({'name': name, 'value': value, 'status': status, 'detail': detail})

    add('Protocol',
        'HTTPS' if features.get('has_https') else 'HTTP',
        'good' if features.get('has_https') else 'bad',
        'Encrypted connection' if features.get('has_https') else 'Unencrypted — credentials visible')

    add('IP in URL',
        'Yes' if features.get('has_ip') else 'No',
        'bad' if features.get('has_ip') else 'good',
        'Domains hide behind IPs in phishing attacks' if features.get('has_ip') else 'Normal domain name used')

    subdomain_count = features.get('subdomain_count', 0)
    add('Subdomain depth',
        str(subdomain_count),
        'bad' if subdomain_count >= 3 else ('warn' if subdomain_count >= 2 else 'good'),
        f'{subdomain_count} subdomain level(s)')

    add('Suspicious TLD',
        'Yes' if features.get('suspicious_tld') else 'No',
        'bad' if features.get('suspicious_tld') else 'good',
        f'TLD: .{features.get("tld", "")}')

    kw_count = features.get('phishing_keyword_count', 0)
    add('Phishing keywords',
        str(kw_count),
        'bad' if kw_count >= 2 else ('warn' if kw_count == 1 else 'good'),
        f'{kw_count} suspicious keyword(s) found in URL')

    url_len = features.get('url_length', 0)
    add('URL length',
        f'{url_len} chars',
        'bad' if url_len > 75 else ('warn' if url_len > 55 else 'good'),
        'Longer URLs are more common in phishing')

    add('@ symbol',
        'Present' if features.get('at_symbol') else 'Absent',
        'bad' if features.get('at_symbol') else 'good',
        'Can be used to disguise the real domain')

    add('Redirect param',
        'Yes' if features.get('has_redirect') else 'No',
        'warn' if features.get('has_redirect') else 'good',
        'URL contains a redirect/goto parameter')

    add('DNS resolves',
        'Yes' if features.get('dns_resolves') else 'No',
        'good' if features.get('dns_resolves') else 'warn',
        'Whether the domain has a valid DNS record')

    domain_age = features.get('domain_age_days', -1)
    if domain_age >= 0:
        age_label = f'{domain_age} days'
        add('Domain age',
            age_label,
            'bad' if domain_age < 30 else ('warn' if domain_age < 180 else 'good'),
            'New domains are high-risk indicators')

    return summary


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    if len(url) > 2000:
        return jsonify({'error': 'URL too long'}), 400

    try:
        metadata = load_model()
        model = metadata['model']
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503

    features = extract_features(url)
    vector = features_to_vector(features)
    X = np.array([vector], dtype=np.float32)

    phishing_prob = float(model.predict_proba(X)[0][1])
    prediction = model.predict(X)[0]

    risk_level = get_risk_level(phishing_prob)
    risk_score = round(phishing_prob * 100, 1)

    feature_summary = build_feature_summary(features)

    verdict_messages = {
        'safe': 'This URL appears legitimate with no major phishing signals detected.',
        'suspicious': 'Some suspicious characteristics found. Proceed with caution.',
        'phishing': 'High phishing risk! Multiple red flags detected — do not enter credentials.',
    }

    response = {
        'url': url,
        'prediction': int(prediction),
        'risk_level': risk_level,
        'risk_score': risk_score,
        'phishing_probability': round(phishing_prob, 4),
        'verdict_message': verdict_messages[risk_level],
        'features': feature_summary,
        'raw_features': {
            k: round(float(v), 4) if isinstance(v, float) else v
            for k, v in features.items()
            if k != 'tld'
        },
    }
    return jsonify(response)


@app.route('/health', methods=['GET'])
def health():
    model_loaded = os.path.exists(MODEL_PATH)
    return jsonify({
        'status': 'ok',
        'model_ready': model_loaded,
        'model_path': MODEL_PATH,
    })


@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception:
        return jsonify({'message': 'Phishing Detector API is running. POST to /predict with {"url": "..."}'}), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting Phishing Detector API...")
    print("  POST http://localhost:5000/predict")
    print("  GET  http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)
