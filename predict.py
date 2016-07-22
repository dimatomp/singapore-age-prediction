from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request


with open('genderWholeClf.dat', 'rb') as f: genderWholeClf = pickle.load(f)
with open('ageTwitterLDAClf.dat', 'rb') as f: ageTwitterLDAClf = pickle.load(f)
with open('genderWholeIndex.dat', 'rb') as f: genderWholeIndex = pickle.load(f)
with open('ageTwitterLDAIndex.dat', 'rb') as f: ageTwitterLDAIndex = pickle.load(f)
with open('ageGroups.dat', 'rb') as f: ageGroups = pickle.load(f)

ageGroups = [next(iter(name for name in ageGroups if ageGroups[name] == i)) for i in range(len(ageGroups))]

app = Flask(__name__)

@app.route('/gender', methods=['POST'])
def predictGender():
    if not request.json or any(col not in request.json for col in genderWholeIndex):
        abort(400)
    data = [[request.json[col] for col in genderWholeIndex]]
    value = genderWholeClf.predict(data)[0]
    if value == 0:
        return 'male'
    elif value == 1:
        return 'female'
    abort(500)

@app.route('/age', methods=['POST'])
def predictAge():
    if not request.json or any(col not in request.json for col in ageTwitterLDAIndex):
        abort(400)
    data = [[request.json[col] for col in ageTwitterLDAIndex]]
    value = ageTwitterLDAClf.predict(data)[0]
    return ageGroups[value]

if __name__ == '__main__':
    app.run(debug=True)
