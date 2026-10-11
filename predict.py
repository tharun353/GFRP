import joblib,pandas as pd
from pathlib import Path
B=Path(__file__).resolve().parent; F=['Glass_Powder_wt%','Glass_Filler_wt%','CGSM','Glass_Fiber_layers','Epoxy_%','Thermal_Aging_hr']; m=joblib.load(B/'model.pkl')
def predict_all(*v):
 p=m.predict(pd.DataFrame([v],columns=F))[0]; return {'Tensile_MPa':float(p[0]),'Impact_kJ_m2':float(p[1]),'Flexural_MPa':float(p[2])}
if __name__=='__main__': print(predict_all(10,10,15,3,35,100))
