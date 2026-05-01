from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
import math

warnings.filterwarnings("ignore")

app = FastAPI(title="Energy Analytics API")

# Allow all origins for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
MONTH_MAP   = {m: i+1 for i, m in enumerate(MONTH_ORDER)}

# Global state to hold models and data
state = {}

def load_data():
    try:
        hh   = pd.read_excel("noida_electricity_household.xlsx")
        comm = pd.read_excel("noida_commercial.xlsx")
    except Exception as e:
        # Create mock data if files don't exist
        print(f"Warning: Could not load excel files ({e}). Using mock data.")
        hh = pd.DataFrame({
            'Month': np.random.choice(MONTH_ORDER, 200),
            'Solar': np.random.choice(['Yes', 'No'], 200),
            'House Type': np.random.choice(['Apartment', 'Independent'], 200),
            'Rooms': np.random.randint(1, 6, 200),
            'Solar Capacity (kW)': np.random.uniform(0, 10, 200),
            'Units Consumed': np.random.uniform(150, 1500, 200),
            'Season': np.random.choice(['Summer', 'Winter', 'Monsoon'], 200)
        })
        comm = pd.DataFrame({
            'Month': np.random.choice(MONTH_ORDER, 130),
            'Solar': np.random.choice(['Yes', 'No'], 130),
            'Business Type': np.random.choice(['Office', 'Retail', 'Restaurant', 'Warehouse'], 130),
            'Connected Load (kW)': np.random.uniform(10, 100, 130),
            'Solar Capacity (kW)': np.random.uniform(0, 50, 130),
            'Units Consumed (After Solar)': np.random.uniform(500, 5000, 130),
            'Season': np.random.choice(['Summer', 'Winter', 'Monsoon'], 130)
        })

    hh['month_num']    = hh['Month'].map(MONTH_MAP)
    hh['solar_flag']   = (hh['Solar'] == 'Yes').astype(int)
    le_h = LabelEncoder()
    hh['house_type_enc'] = le_h.fit_transform(hh['House Type'])

    comm['month_num']   = comm['Month'].map(MONTH_MAP)
    comm['solar_flag']  = (comm['Solar'] == 'Yes').astype(int)
    le_c = LabelEncoder()
    comm['business_enc'] = le_c.fit_transform(comm['Business Type'])

    return hh, comm, le_h, le_c

def train_models(hh, comm):
    results = {}

    feat_h = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
    X_h = hh[feat_h].values
    y_h = hh['Units Consumed'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

    lr_h  = LinearRegression().fit(X_tr, y_tr)
    gb_h  = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, subsample=0.8, random_state=42).fit(X_tr, y_tr)

    results['hh'] = {
        'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te,
        'lr': lr_h, 'gb': gb_h,
        'fi': gb_h.feature_importances_.tolist(), 'feat_names': feat_h,
        'lr_pred': lr_h.predict(X_te).tolist(),
        'gb_pred': gb_h.predict(X_te).tolist(),
    }

    feat_c = ['month_num','business_enc','solar_flag','Connected Load (kW)','Solar Capacity (kW)']
    X_c = comm[feat_c].values
    y_c = comm['Units Consumed (After Solar)'].values
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

    lr_c  = LinearRegression().fit(Xc_tr, yc_tr)
    gb_c  = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, random_state=42).fit(Xc_tr, yc_tr)

    results['comm'] = {
        'X_tr': Xc_tr, 'X_te': Xc_te, 'y_tr': yc_tr, 'y_te': yc_te,
        'lr': lr_c, 'gb': gb_c,
        'fi': gb_c.feature_importances_.tolist(), 'feat_names': feat_c,
        'lr_pred': lr_c.predict(Xc_te).tolist(),
        'gb_pred': gb_c.predict(Xc_te).tolist(),
    }

    return results

@app.on_event("startup")
async def startup_event():
    hh, comm, le_h, le_c = load_data()
    models = train_models(hh, comm)
    state['hh'] = hh
    state['comm'] = comm
    state['le_h'] = le_h
    state['le_c'] = le_c
    state['models'] = models

@app.get("/api/overview")
async def get_overview():
    hh = state['hh']
    comm = state['comm']
    avg_hh = hh['Units Consumed'].mean()
    avg_comm = comm['Units Consumed (After Solar)'].mean()
    solar_pct = (hh['Solar'] == 'Yes').mean() * 100
    
    return {
        "avg_hh_kwh": round(avg_hh),
        "avg_comm_kwh": round(avg_comm),
        "solar_adoption_pct": round(solar_pct),
        "street_lights_kwh": 36217
    }

class PredictRequest(BaseModel):
    rooms: int
    solar: str
    solar_kw: float
    house_type: str
    month: str
    dataset: str

@app.post("/api/predict")
async def predict(req: PredictRequest):
    if req.dataset == "Household":
        le = state['le_h']
        model_gb = state['models']['hh']['gb']
        model_lr = state['models']['hh']['lr']
        try:
            ht_enc = int(le.transform([req.house_type])[0])
        except ValueError:
            ht_enc = 0
            
        inp = [[MONTH_MAP.get(req.month, 1), 1 if req.solar == 'Yes' else 0, ht_enc, req.rooms, req.solar_kw]]
    else:
        # Mock logic for commercial if dataset == 'Commercial'
        le = state['le_c']
        model_gb = state['models']['comm']['gb']
        model_lr = state['models']['comm']['lr']
        try:
            bt_enc = int(le.transform([req.house_type])[0]) # using house_type as business type
        except ValueError:
            bt_enc = 0
        inp = [[MONTH_MAP.get(req.month, 1), bt_enc, 1 if req.solar == 'Yes' else 0, req.rooms * 10, req.solar_kw]] # dummy logic for comm
        
    pred_gb = max(0.0, float(model_gb.predict(inp)[0]))
    pred_lr = max(0.0, float(model_lr.predict(inp)[0]))
    
    # model scores for the right pane
    r2_gb = r2_score(state['models']['hh']['y_te'], state['models']['hh']['gb_pred']) if req.dataset == "Household" else r2_score(state['models']['comm']['y_te'], state['models']['comm']['gb_pred'])
    r2_lr = r2_score(state['models']['hh']['y_te'], state['models']['hh']['lr_pred']) if req.dataset == "Household" else r2_score(state['models']['comm']['y_te'], state['models']['comm']['lr_pred'])
    
    return {
        "prediction_kwh": round(pred_gb, 1),
        "prediction_lr_kwh": round(pred_lr, 1),
        "model_scores": {
            "xgboost_r2": round(r2_gb, 4),
            "linear_r2": round(r2_lr, 4)
        }
    }

class ForecastRequest(BaseModel):
    base_kwh: float
    years: int
    pop_growth: float
    urbanization: float
    income_growth: float
    ev_adoption: bool
    solar_growth: bool

@app.post("/api/forecast")
async def get_forecast(req: ForecastRequest):
    years = req.years
    pop_growth = req.pop_growth / 100.0
    urbanization = req.urbanization / 100.0
    income_growth = req.income_growth / 100.0
    ev_impact = 1.02 if req.ev_adoption else 1.0
    solar_impact = 0.98 if req.solar_growth else 1.0
    
    projections = []
    
    for y in range(1, 51):
        income_factor = (1 + income_growth)**(y * 0.3)
        val = req.base_kwh * ((1 + pop_growth)**y) * ((1 + urbanization)**(y*0.5)) * income_factor * (ev_impact**y) * (solar_impact**y)
        if y <= years:
            projections.append({
                "year": y,
                "kwh": round(val, 1),
                "low": round(val * 0.85, 1),
                "high": round(val * 1.15, 1)
            })
    return projections

@app.get("/api/monthly")
async def get_monthly():
    hh_m = state['hh'].groupby('month_num')['Units Consumed'].mean().sort_index()
    comm_m = state['comm'].groupby('month_num')['Units Consumed (After Solar)'].mean().sort_index()
    return {
        "months": [MONTH_ORDER[i-1][:3] for i in hh_m.index],
        "household": hh_m.round(1).tolist(),
        "commercial": comm_m.round(1).tolist()
    }

@app.get("/api/seasonal")
async def get_seasonal():
    hh_s = state['hh'].groupby('Season')['Units Consumed'].mean().round(1)
    comm_s = state['comm'].groupby('Season')['Units Consumed (After Solar)'].mean().round(1)
    return {
        "seasons": hh_s.index.tolist(),
        "household": hh_s.tolist(),
        "commercial": comm_s.tolist()
    }

@app.get("/api/commercial")
async def get_commercial():
    biz = state['comm'].groupby('Business Type')['Units Consumed (After Solar)'].mean().sort_values().round(1)
    return {
        "types": biz.index.tolist(),
        "kwh": biz.tolist()
    }

@app.get("/api/features")
async def get_features():
    fi_vals = state['models']['hh']['fi']
    fi_names = state['models']['hh']['feat_names']
    
    # Sort
    combined = sorted(zip(fi_names, fi_vals), key=lambda x: x[1])
    
    return {
        "features": [x[0] for x in combined],
        "importance": [round(x[1]*100, 1) for x in combined]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
