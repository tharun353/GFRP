# GFRP Composite Machine Learning Prediction System — 3 Outputs

Six inputs are used by a multi-output RandomForestRegressor to predict:
- Tensile_MPa (MPa)
- Impact_kJ_m2 (kJ/m²)
- Flexural_MPa (MPa)

Inputs: Glass_Powder_wt%, Glass_Filler_wt%, CGSM, Glass_Fiber_layers, Epoxy_%, Thermal_Aging_hr.

## Run
pip install -r requirements.txt
python train_model.py
python app.py

Open http://127.0.0.1:5000

The included `dummy samples.xlsx` is synthetic demonstration data. Replace it with real experimental data for research use while preserving the exact column names. Included metrics are not research results.
