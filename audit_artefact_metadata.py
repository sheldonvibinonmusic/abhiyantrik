import joblib

data = joblib.load('model_artefact.joblib')
print("Keys in model_artefact.joblib:", list(data.keys()))
print("Threshold:", data['threshold'])
print("Features count:", len(data['features']))
print("Features:", data['features'])
print("Categorical columns:", data['cat_cols'])
print("Nominal centers:", data['nominal_centers'])
print("Nominal scales:", data['nominal_scales'])
print("Model 1 type:", type(data['hgb_reg']))
print("Model 2 type:", type(data['hgb_weighted']))
print("Model 1 params:", data['hgb_reg'].get_params())
print("Model 2 params:", data['hgb_weighted'].get_params())
