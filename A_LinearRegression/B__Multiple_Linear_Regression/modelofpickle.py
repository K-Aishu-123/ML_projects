import pickle
with open('model.pkl','rb') as f:
    arun=pickle.load(f)
arun.predict([[230.1,37.8,69.2]])