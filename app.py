import joblib
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# LOAD ALL MODELS
print("Loading system modules...")

# Load ML Components (Priority)
rf_model = joblib.load('priority_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load DL Components (Department)
dl_model = tf.keras.models.load_model('department_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

print("✅ System Online.")

# DEFINE THE PREDICTION FUNCTION
def process_ticket(ticket_text):
    print(f"\n📨 NEW TICKET: '{ticket_text}'")
    
    # --- TASK A: PREDICT PRIORITY (Machine Learning) ---
    # 1. Vectorize
    text_tfidf = tfidf_vectorizer.transform([ticket_text])
    # 2. Predict
    priority_pred = rf_model.predict(text_tfidf)[0]
    
    # --- TASK B: PREDICT DEPARTMENT (Deep Learning) ---
    # 1. Tokenize & Pad
    seq = tokenizer.texts_to_sequences([ticket_text])
    padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
    # 2. Predict (returns probabilities)
    dept_prob = dl_model.predict(padded, verbose=0)
    # 3. Get highest probability class
    dept_index = np.argmax(dept_prob)
    department_pred = label_encoder.inverse_transform([dept_index])[0]
    confidence = np.max(dept_prob) * 100
    
    # --- REPORT ---
    print("-" * 30)
    print(f"🚨 Priority:    {priority_pred.upper()}")
    print(f"🏢 Department:  {department_pred} ({confidence:.1f}% confidence)")
    print("-" * 30)

# TEST IT LIVE!
# Let's throw some fake scenarios at your bot
process_ticket("My internet is completely down and I have a meeting in 5 minutes! Fix it now!")
process_ticket("I want to return my order, the shoes are too small.")
process_ticket("How do I update the firmware on my Xbox?")
process_ticket("Can you check the status of my flight DL123?")