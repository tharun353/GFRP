from pathlib import Path
import pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
BASE_DIR=Path(__file__).resolve().parent
F=['Glass_Powder_wt%','Glass_Filler_wt%','CGSM','Glass_Fiber_layers','Epoxy_%','Thermal_Aging_hr']; T=['Tensile_MPa','Impact_kJ_m2','Flexural_MPa']
df=pd.read_excel(BASE_DIR/'dummy samples.xlsx')[F+T].dropna(); Xtr,Xte,Ytr,Yte=train_test_split(df[F],df[T],test_size=.2,random_state=42)
m=RandomForestRegressor(n_estimators=300,random_state=42,min_samples_leaf=2,n_jobs=-1); m.fit(Xtr,Ytr); p=m.predict(Xte); joblib.dump(m,BASE_DIR/'model.pkl')
for i,t in enumerate(T): print(f'{t}: R2={r2_score(Yte[t],p[:,i]):.4f}, MAE={mean_absolute_error(Yte[t],p[:,i]):.4f}')
print('Saved model.pkl')