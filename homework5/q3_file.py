import pickle
import os

input_file = f'pipeline_v1.bin'


with open(input_file, 'rb') as f_in:
   dv, model = pickle.load(f_in)


check = {"lead_source": "paid_ads", "number_of_courses_viewed": 2, "annual_income": 79276.0}

X = dv.transform([check])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)


