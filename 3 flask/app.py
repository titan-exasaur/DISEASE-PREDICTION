from flask import *
import os,sys
app = Flask(__name__)

user_id='admin'
user_pwd='admin123'



@app.route('/')
def login():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

# @app.route('/eda')
# def eda():
#     return render_template('eda.html')
    

@app.route('/detector')
def detector():
    return render_template('detector.html')

@app.route('/model_parameter')
def model_parameter():
    return render_template('model_parameter.html')

# @app.route('/display/<filename>')
# def display_image(filename):
#     # print('display_image filename: ' + filename)
#     return redirect(url_for('static_new', filename='uploads/' + filename))

@app.route('/submit_detector', methods=['POST'])
def choose_file():
    if request.method == 'POST':
        itching = int(request.form.get('itching'))
        skin_rash = int(request.form.get('skin_rash'))
        nodal_skin_eruptions = int(request.form.get('nodal_skin_eruptions'))
        continuous_sneezing = int(request.form.get('continuous_sneezing'))
        chills = int(request.form.get('chills'))
        joint_pain = int(request.form.get('joint_pain'))
        stomach_pain = int(request.form.get('stomach_pain'))
        acidity = int(request.form.get('acidity'))
        muscle_wasting = int(request.form.get('muscle_wasting'))
        vomiting = int(request.form.get('vomiting'))
        burning_micturition = int(request.form.get('burning_micturition'))
        fatigue = int(request.form.get('fatigue'))
        weight_gain = int(request.form.get('weight_gain'))
        anxiety = int(request.form.get('anxiety'))
        weight_loss = int(request.form.get('weight_loss'))
        cough = int(request.form.get('cough'))
        sunken_eyes = int(request.form.get('sunken_eyes'))
        indigestion = int(request.form.get('indigestion'))
        headache = int(request.form.get('headache'))
        yellowish_skin = int(request.form.get('yellowish_skin'))
        pain_behind_the_eyes = int(request.form.get('pain_behind_the_eyes'))
        constipation = int(request.form.get('constipation'))
        diarrhoea = int(request.form.get('diarrhoea'))
        yellow_urine = int(request.form.get('yellow_urine'))
        acute_liver_failure = int(request.form.get('acute_liver_failure'))
        swelling_of_stomach = int(request.form.get('swelling_of_stomach'))
        cramps = int(request.form.get('cramps'))
        spinning_movements = int(request.form.get('spinning_movements'))
        weakness_of_one_body_side = int(request.form.get('weakness_of_one_body_side'))
        family_history = int(request.form.get('family_history'))
        pus_filled_pimples = int(request.form.get('pus_filled_pimples'))
        skin_peeling = int(request.form.get('skin_peeling'))
        blister = int(request.form.get('blister'))
         
        # Importing necessary libraries
        import pandas as pd
        import numpy as np
        from sklearn.metrics import classification_report
        import warnings

        warnings.filterwarnings('ignore')

        # LOADING THE DATASET & PRE - PROCESSING
            
        # Mapping target values
        prognosis_num_to_text = {0: 'Fungal infection', 1: 'Allergy', 2: 'GERD',
                                3: 'Chronic cholestasis', 4: 'Drug Reaction', 5: 'Peptic ulcer diseae',
                                6: 'AIDS', 7: 'Diabetes ', 8: 'Gastroenteritis',
                                9: 'Bronchial Asthma', 10: 'Hypertension ', 11: 'Migraine',
                                12: 'Cervical spondylosis', 13: 'Paralysis (brain hemorrhage)', 14: 'Jaundice',
                                15: 'Malaria', 16: 'Chicken pox', 17: 'Dengue', 18: 'Typhoid',
                                19: 'hepatitis A', 20: 'Hepatitis B', 21: 'Hepatitis C',
                                22: 'Hepatitis D', 23: 'Hepatitis E', 24: 'Alcoholic hepatitis',
                                25: 'Tuberculosis', 26: 'Common Cold', 27: 'Pneumonia',
                                28: 'Dimorphic hemmorhoids(piles)', 29: 'Heart attack', 30: 'Varicose veins',
                                31: 'Hypothyroidism', 32: 'Hyperthyroidism', 33: 'Hypoglycemia',
                                34: 'Osteoarthristis', 35: 'Arthritis', 36: '(vertigo) Paroymsal  Positional Vertigo',
                                37: 'Acne', 38: 'Urinary tract infection', 39: 'Psoriasis', 40: 'Impetigo'}




        new_user_input = [itching, skin_rash, nodal_skin_eruptions, continuous_sneezing,
            chills, joint_pain, stomach_pain, acidity, muscle_wasting,
            vomiting, burning_micturition, fatigue, weight_gain, anxiety,
            weight_loss, cough, sunken_eyes, indigestion, headache, yellowish_skin,
            pain_behind_the_eyes, constipation, diarrhoea, yellow_urine,
            acute_liver_failure, swelling_of_stomach, cramps,
            spinning_movements, weakness_of_one_body_side, family_history,
            pus_filled_pimples, skin_peeling, blister]

        new_user_input = np.array([new_user_input])
        print(type(new_user_input))


        from tensorflow.keras.models import load_model
        ann_model = load_model('/home/kumar/Downloads/24 [BMSIT] DISEASE PREDICTION/3 flask/ann_model.h5')

        probabilities = ann_model.predict(new_user_input)[0]

        # probabilities = ann_model.predict(np.array([new_user_input]))[0]
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_diseases = [prognosis_num_to_text[idx] for idx in top_indices]
        top_probabilities = probabilities[top_indices]

        # display these
        # print("Top 3 Predictions:")
        # print(f"{top_diseases[0]}:{top_probabilities[0]*100}%")
        # print(f"{top_diseases[1]}:{top_probabilities[1]*100}%")
        # print(f"{top_diseases[2]}:{top_probabilities[2]*100}%")

        top_diseases_one = f"{top_diseases[0]}"





        return render_template('detector.html',top_diseases_one =top_diseases_one)

if __name__=='__main__':
    app.run(debug=True)