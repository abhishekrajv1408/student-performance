from flask import  Flask,request,jsonify
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods=['POST'])
def home():
    GENERAL_APPEARANCE = request.form.get('GENERAL_APPEARANCE')
    MENTAL_ALERTNESS = request.form.get('MENTAL_ALERTNESS')
    MANNER_OF_SPEAKING = request.form.get('MANNER_OF_SPEAKING')
    PHYSICAL_CONDITION = request.form.get('PHYSICAL_CONDITION')
    SELF_CONFIDENCEE = request.form.get('SELF_CONFIDENCEE')
    ABILITY_TO_PRESENT_IDEAS = request.form.get('ABILITY_TO_PRESENT_IDEAS')
    COMMUNICATION_SKILLS = request.form.get('COMMUNICATION_SKILLS')
    # result={'GENERAL_APPEARANCE':GENERAL_APPEARANCE,'MANNER_OF_SPEAKING':MANNER_OF_SPEAKING,'PHYSICAL_CONDITION':PHYSICAL_CONDITION,'SELF_CONFIDENCEE':SELF_CONFIDENCEE,'COMMUNICATION_SKILLS':COMMUNICATION_SKILLS}
    input=np.array([[GENERAL_APPEARANCE,MANNER_OF_SPEAKING,MENTAL_ALERTNESS,PHYSICAL_CONDITION,SELF_CONFIDENCEE,ABILITY_TO_PRESENT_IDEAS,COMMUNICATION_SKILLS]])
    result=model.predict(input)
    if result==1:
        s="Employed"
    else:
        s="Unemployed"
    return jsonify(s)

if __name__ == '__main__':
    app.run(debug=True)




    