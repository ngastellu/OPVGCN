import joblib

m1 = joblib.load('PCE_model')
m2 = joblib.load('lgb_model')

print(m1 == m2)

