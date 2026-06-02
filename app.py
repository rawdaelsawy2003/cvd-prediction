"""
app.py — Cardiovascular Disease Prediction Web App
===================================================
Run:  python app.py
URL:  http://127.0.0.1:5000

Requirements:
    pip install flask scikit-learn joblib numpy
"""

from flask import Flask, render_template_string, request, jsonify
import joblib
import numpy as np
import os

# ── Load saved artifacts ──────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH   = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'features.pkl')

model    = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)

app = Flask(__name__)

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CardioRisk — CVD Prediction</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --red:    #E63946;
      --navy:   #1D3557;
      --blue:   #457B9D;
      --light:  #A8DADC;
      --cream:  #F1FAEE;
      --white:  #FFFFFF;
      --shadow: 0 8px 32px rgba(29,53,87,0.12);
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'DM Sans', sans-serif;
      background: linear-gradient(135deg, #1D3557 0%, #457B9D 60%, #A8DADC 100%);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem 4rem;
    }

    .hero {
      text-align: center;
      color: var(--white);
      margin-bottom: 2.5rem;
      animation: fadeDown 0.7s ease both;
    }
    .hero h1 {
      font-family: 'DM Serif Display', serif;
      font-size: clamp(2rem, 5vw, 3.2rem);
      letter-spacing: -0.5px;
    }
    .hero p {
      font-size: 1.05rem;
      opacity: 0.85;
      margin-top: 0.5rem;
      font-weight: 300;
    }
    .heart-icon {
      font-size: 3rem;
      animation: pulse 1.5s ease-in-out infinite;
      display: block;
      margin-bottom: 0.5rem;
    }

    .card {
      background: var(--white);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 2.5rem 2rem;
      width: 100%;
      max-width: 760px;
      animation: fadeUp 0.7s 0.2s ease both;
    }

    .section-title {
      font-family: 'DM Serif Display', serif;
      font-size: 1.1rem;
      color: var(--navy);
      margin: 1.8rem 0 1rem;
      padding-bottom: 0.4rem;
      border-bottom: 2px solid var(--light);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .section-title:first-of-type { margin-top: 0; }

    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
    @media (max-width: 520px) { .grid { grid-template-columns: 1fr; } }

    .field { display: flex; flex-direction: column; gap: 0.35rem; }
    label {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--navy);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    input, select {
      padding: 0.65rem 0.9rem;
      border: 1.5px solid #d0d8e4;
      border-radius: 10px;
      font-family: 'DM Sans', sans-serif;
      font-size: 0.95rem;
      color: var(--navy);
      background: #f8fafc;
      transition: border-color 0.2s, box-shadow 0.2s;
      outline: none;
    }
    input:focus, select:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(69,123,157,0.15);
      background: #fff;
    }

    .btn {
      margin-top: 2rem;
      width: 100%;
      padding: 1rem;
      background: linear-gradient(135deg, var(--red), #c1121f);
      color: white;
      font-family: 'DM Sans', sans-serif;
      font-size: 1.05rem;
      font-weight: 600;
      border: none;
      border-radius: 14px;
      cursor: pointer;
      letter-spacing: 0.03em;
      transition: transform 0.15s, box-shadow 0.2s;
      box-shadow: 0 4px 16px rgba(230,57,70,0.35);
    }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(230,57,70,0.4); }
    .btn:active { transform: translateY(0); }

    /* Result panel */
    #result {
      display: none;
      margin-top: 1.8rem;
      border-radius: 16px;
      padding: 1.5rem 2rem;
      text-align: center;
      animation: fadeUp 0.4s ease both;
    }
    #result.high {
      background: linear-gradient(135deg, #fff0f1, #ffe0e2);
      border: 2px solid var(--red);
    }
    #result.low {
      background: linear-gradient(135deg, #f0fafb, #dff2f3);
      border: 2px solid var(--light);
    }
    #result .result-icon { font-size: 2.5rem; }
    #result h2 {
      font-family: 'DM Serif Display', serif;
      font-size: 1.6rem;
      margin: 0.5rem 0 0.3rem;
    }
    #result.high h2 { color: var(--red); }
    #result.low h2  { color: #2a7a5f; }
    #result p { font-size: 0.95rem; color: #444; }

    .prob-bar-wrap { margin-top: 1rem; }
    .prob-label { font-size: 0.8rem; font-weight: 600; color: #555; margin-bottom: 0.3rem; display: flex; justify-content: space-between; }
    .prob-bar { background: #e8ecf0; border-radius: 20px; height: 10px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 20px; transition: width 1s cubic-bezier(.4,0,.2,1); }

    .disclaimer {
      font-size: 0.72rem;
      color: #999;
      margin-top: 1rem;
      font-style: italic;
    }

    footer {
      color: rgba(255,255,255,0.6);
      font-size: 0.8rem;
      margin-top: 2rem;
      text-align: center;
    }

    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50%       { transform: scale(1.12); }
    }
    @keyframes fadeDown {
      from { opacity: 0; transform: translateY(-20px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(20px); }
      to   { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>

<div class="hero">
  <span class="heart-icon">🫀</span>
  <h1>CardioRisk Predictor</h1>
  <p>Enter your clinical measurements to estimate cardiovascular disease risk</p>
</div>

<div class="card">

  <p class="section-title">👤 Personal Info</p>
  <div class="grid">
    <div class="field">
      <label>Age (years)</label>
      <input type="number" id="age_years" min="30" max="65" placeholder="e.g. 50" required>
    </div>
    <div class="field">
      <label>Gender</label>
      <select id="gender">
        <option value="1">Female</option>
        <option value="2">Male</option>
      </select>
    </div>
    <div class="field">
      <label>Height (cm)</label>
      <input type="number" id="height" min="100" max="220" placeholder="e.g. 170" required>
    </div>
    <div class="field">
      <label>Weight (kg)</label>
      <input type="number" id="weight" min="30" max="200" placeholder="e.g. 75" required>
    </div>
  </div>

  <p class="section-title">🩺 Clinical Measurements</p>
  <div class="grid">
    <div class="field">
      <label>Systolic BP (ap_hi)</label>
      <input type="number" id="ap_hi" min="60" max="250" placeholder="e.g. 120" required>
    </div>
    <div class="field">
      <label>Diastolic BP (ap_lo)</label>
      <input type="number" id="ap_lo" min="40" max="200" placeholder="e.g. 80" required>
    </div>
    <div class="field">
      <label>Cholesterol Level</label>
      <select id="cholesterol">
        <option value="1">Normal</option>
        <option value="2">Above Normal</option>
        <option value="3">Well Above Normal</option>
      </select>
    </div>
    <div class="field">
      <label>Glucose Level</label>
      <select id="gluc">
        <option value="1">Normal</option>
        <option value="2">Above Normal</option>
        <option value="3">Well Above Normal</option>
      </select>
    </div>
  </div>

  <p class="section-title">🏃 Lifestyle</p>
  <div class="grid">
    <div class="field">
      <label>Do you smoke?</label>
      <select id="smoke">
        <option value="0">No</option>
        <option value="1">Yes</option>
      </select>
    </div>
    <div class="field">
      <label>Alcohol intake?</label>
      <select id="alco">
        <option value="0">No</option>
        <option value="1">Yes</option>
      </select>
    </div>
    <div class="field">
      <label>Physically active?</label>
      <select id="active">
        <option value="1">Yes</option>
        <option value="0">No</option>
      </select>
    </div>
  </div>

  <button class="btn" onclick="predict()">🔍 Predict CVD Risk</button>

  <div id="result">
    <div class="result-icon" id="r-icon"></div>
    <h2 id="r-title"></h2>
    <p id="r-desc"></p>
    <div class="prob-bar-wrap">
      <div class="prob-label">
        <span>CVD Probability</span>
        <span id="r-prob-pct"></span>
      </div>
      <div class="prob-bar">
        <div class="prob-fill" id="r-prob-fill"></div>
      </div>
    </div>
    <p class="disclaimer">⚠️ This tool is for educational purposes only and is NOT a substitute for medical advice.</p>
  </div>

</div>

<footer>Final Data Science Project &nbsp;|&nbsp; Cardiovascular Disease Prediction &nbsp;|&nbsp; Random Forest Classifier</footer>

<script>
  async function predict() {
    const age_years    = parseFloat(document.getElementById('age_years').value);
    const gender       = parseInt(document.getElementById('gender').value);
    const height       = parseFloat(document.getElementById('height').value);
    const weight       = parseFloat(document.getElementById('weight').value);
    const ap_hi        = parseFloat(document.getElementById('ap_hi').value);
    const ap_lo        = parseFloat(document.getElementById('ap_lo').value);
    const cholesterol  = parseInt(document.getElementById('cholesterol').value);
    const gluc         = parseInt(document.getElementById('gluc').value);
    const smoke        = parseInt(document.getElementById('smoke').value);
    const alco         = parseInt(document.getElementById('alco').value);
    const active       = parseInt(document.getElementById('active').value);

    if (!age_years || !height || !weight || !ap_hi || !ap_lo) {
      alert('Please fill in all numeric fields.');
      return;
    }
    if (ap_hi < ap_lo) {
      alert('Systolic BP must be ≥ Diastolic BP.');
      return;
    }

    const btn = document.querySelector('.btn');
    btn.textContent = '⏳ Analysing...';
    btn.disabled = true;

    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ age_years, gender, height, weight,
                             ap_hi, ap_lo, cholesterol, gluc,
                             smoke, alco, active })
    });
    const data = await response.json();

    btn.textContent = '🔍 Predict CVD Risk';
    btn.disabled = false;

    const resultDiv = document.getElementById('result');
    const prob = data.probability;
    const pct  = (prob * 100).toFixed(1);

    resultDiv.className = prob >= 0.5 ? 'high' : 'low';
    document.getElementById('r-icon').textContent  = prob >= 0.5 ? '⚠️' : '✅';
    document.getElementById('r-title').textContent = prob >= 0.5
      ? 'High CVD Risk Detected'
      : 'Low CVD Risk';
    document.getElementById('r-desc').textContent  = prob >= 0.5
      ? 'The model suggests elevated cardiovascular risk. Please consult a physician.'
      : 'The model suggests a lower risk of cardiovascular disease. Keep up healthy habits!';
    document.getElementById('r-prob-pct').textContent = pct + '%';

    const fill = document.getElementById('r-prob-fill');
    fill.style.width  = '0%';
    fill.style.background = prob >= 0.5
      ? 'linear-gradient(90deg,#E63946,#c1121f)'
      : 'linear-gradient(90deg,#2a9d8f,#52b788)';
    resultDiv.style.display = 'block';
    setTimeout(() => { fill.style.width = pct + '%'; }, 80);
  }
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Derive engineered features
    age_years     = float(data['age_years'])
    height        = float(data['height'])
    weight        = float(data['weight'])
    ap_hi         = float(data['ap_hi'])
    ap_lo         = float(data['ap_lo'])
    cholesterol   = int(data['cholesterol'])
    gluc          = int(data['gluc'])
    smoke         = int(data['smoke'])
    alco          = int(data['alco'])
    active        = int(data['active'])
    gender        = int(data['gender'])

    bmi           = round(weight / (height / 100) ** 2, 2)
    pulse_pressure = ap_hi - ap_lo
    hypertension  = 1 if (ap_hi >= 140 or ap_lo >= 90) else 0
    obese         = 1 if bmi >= 30 else 0

    # Build feature vector in exact order used during training
    feature_map = {
        'age_years':     age_years,
        'gender':        gender,
        'height':        height,
        'weight':        weight,
        'bmi':           bmi,
        'ap_hi':         ap_hi,
        'ap_lo':         ap_lo,
        'pulse_pressure':pulse_pressure,
        'cholesterol':   cholesterol,
        'gluc':          gluc,
        'smoke':         smoke,
        'alco':          alco,
        'active':        active,
        'hypertension':  hypertension,
        'obese':         obese,
    }

    input_values = np.array([[feature_map[f] for f in features]])
    input_scaled = scaler.transform(input_values)

    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return jsonify({
        'prediction':  int(prediction),
        'probability': float(round(probability, 4)),
        'risk_label':  'High Risk' if prediction == 1 else 'Low Risk'
    })


if __name__ == '__main__':
    app.run(debug=True)
