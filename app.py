from flask import Flask, render_template, request, jsonify
import joblib, pandas as pd
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent
model=joblib.load(BASE_DIR/'model.pkl')
FEATURES=['Glass_Powder_wt%','Glass_Filler_wt%','CGSM','Glass_Fiber_layers','Epoxy_%','Thermal_Aging_hr']
app=Flask(__name__)
@app.route('/')
def index(): return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    try:
        d=request.get_json(force=True); vals=[float(d[f]) for f in FEATURES]
        if vals[3]<1 or vals[5]<0: return jsonify(error='Glass Fiber Layers must be >=1 and Thermal Aging Hours cannot be negative.'),400
        p=model.predict(pd.DataFrame([vals],columns=FEATURES))[0]
        return jsonify(success=True,predictions={'tensile':round(float(p[0]),2),'impact':round(float(p[1]),2),'flexural':round(float(p[2]),2)})
    except Exception as e: return jsonify(error=str(e)),400
if __name__=='__main__': app.run(host='127.0.0.1',port=5000,debug=True)
