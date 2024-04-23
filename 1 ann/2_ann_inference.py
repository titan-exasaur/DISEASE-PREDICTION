# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

# LOADING THE DATASET & PRE - PROCESSING
df = pd.read_csv('multi_disease.csv')

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

#binod these are inputs, total 33 inputs
itching = int(input("Do you have itching? Enter 1 for yes, 0 for no: "))
skin_rash = int(input("Do you have a skin rash? Enter 1 for yes, 0 for no: "))
nodal_skin_eruptions = int(input("Do you have nodal skin eruptions? Enter 1 for yes, 0 for no: "))
continuous_sneezing = int(input("Do you have continuous sneezing? Enter 1 for yes, 0 for no: "))
chills = int(input("Do you have chills? Enter 1 for yes, 0 for no: "))
joint_pain = int(input("Do you have joint pain? Enter 1 for yes, 0 for no: "))
stomach_pain = int(input("Do you have stomach pain? Enter 1 for yes, 0 for no: "))
acidity = int(input("Do you have acidity? Enter 1 for yes, 0 for no: "))
muscle_wasting = int(input("Do you have muscle wasting? Enter 1 for yes, 0 for no: "))
vomiting = int(input("Do you have vomiting? Enter 1 for yes, 0 for no: "))
burning_micturition = int(input("Do you have burning micturition? Enter 1 for yes, 0 for no: "))
fatigue = int(input("Do you have fatigue? Enter 1 for yes, 0 for no: "))
weight_gain = int(input("Do you have weight gain? Enter 1 for yes, 0 for no: "))
anxiety = int(input("Do you have anxiety? Enter 1 for yes, 0 for no: "))
weight_loss = int(input("Do you have weight loss? Enter 1 for yes, 0 for no: "))
cough = int(input("Do you have cough? Enter 1 for yes, 0 for no: "))
sunken_eyes = int(input("Do you have sunken eyes? Enter 1 for yes, 0 for no: "))
indigestion = int(input("Do you have indigestion? Enter 1 for yes, 0 for no: "))
headache = int(input("Do you have headache? Enter 1 for yes, 0 for no: "))
yellowish_skin = int(input("Do you have yellowish skin? Enter 1 for yes, 0 for no: "))
pain_behind_the_eyes = int(input("Do you have pain behind the eyes? Enter 1 for yes, 0 for no: "))
constipation = int(input("Do you have constipation? Enter 1 for yes, 0 for no: "))
diarrhoea = int(input("Do you have diarrhoea? Enter 1 for yes, 0 for no: "))
yellow_urine = int(input("Is the urine dark yellow? Enter 1 for yes, 0 for no: "))
acute_liver_failure = int(input("Do you have acute liver failure? Enter 1 for yes, 0 for no: "))
swelling_of_stomach = int(input("Do you have swelling of stomach? Enter 1 for yes, 0 for no: "))
cramps = int(input("Do you have cramps? Enter 1 for yes, 0 for no: "))
spinning_movements = int(input("Do you have spinning movements? Enter 1 for yes, 0 for no: "))
weakness_of_one_body_side = int(input("Do you have weakness of one body side? Enter 1 for yes, 0 for no: "))
family_history = int(input("Do you have family history? Enter 1 for yes, 0 for no: "))
pus_filled_pimples = int(input("Do you have pus-filled pimples? Enter 1 for yes, 0 for no: "))
skin_peeling = int(input("Do you have skin peeling? Enter 1 for yes, 0 for no: "))
blister = int(input("Do you have a blister? Enter 1 for yes, 0 for no: "))


new_user_input = [itching, skin_rash, nodal_skin_eruptions, continuous_sneezing,
       chills, joint_pain, stomach_pain, acidity, muscle_wasting,
       vomiting, burning_micturition, fatigue, weight_gain, anxiety,
       weight_loss, cough, sunken_eyes, indigestion, headache, yellowish_skin,
       pain_behind_the_eyes, constipation, diarrhoea, yellow_urine,
       acute_liver_failure, swelling_of_stomach, cramps,
       spinning_movements, weakness_of_one_body_side, family_history,
       pus_filled_pimples, skin_peeling, blister]


from tensorflow.keras.models import load_model
ann_model = load_model('ann_model.h5')

new_user_input = np.array([new_user_input])  # Convert to numpy array and add an extra dimension
probabilities = ann_model.predict(new_user_input)[0]

probabilities = ann_model.predict(np.array(new_user_input))[0]
top_indices = np.argsort(probabilities)[::-1][:3]
top_diseases = [prognosis_num_to_text[idx] for idx in top_indices]
top_probabilities = probabilities[top_indices]

#binod display these
print("Top 3 Predictions:")
print(f"{top_diseases[0]}:{top_probabilities[0]*100}%")
print(f"{top_diseases[1]}:{top_probabilities[1]*100}%")
print(f"{top_diseases[2]}:{top_probabilities[2]*100}%")


import matplotlib.pyplot as plt
# Data for plotting
labels = top_diseases
sizes = top_probabilities
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.1, 0, 0)  # explode the 1st slice

# Plot
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Top 3 Predicted Diseases with Probabilities')
plt.savefig('static_new/uploads/results.png',bbox_inches='tight')#binod display this also
plt.show()
