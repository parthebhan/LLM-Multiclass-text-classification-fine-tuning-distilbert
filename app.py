import streamlit as st
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define model path (directory containing the TensorFlow weights)
model_path = r'F:\mlproject\LLM\LLM_Text_Classification_DistilBERT_uncased'

# Load the model
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

# Define prediction mapping
prediction_mapping = {
    0: 'business',
    1: 'entertainment',
    2: 'politics',
    3: 'sport',
    4: 'tech'
}

# Function to predict using loaded tokenizer and model
def predict(input_text):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors='tf', max_length=512, truncation=True, padding=True)
    
    # Predict
    logits = model(inputs)[0]
    prediction = tf.argmax(logits, axis=1).numpy()[0]
    
    return prediction

# Streamlit app
def main():
    st.title('Multiclass Text Classification')
    st.write('Model - DistilBERT-base-uncased')
    
    st.write('Classification Categories  : business, entertainment, politics, sport, tech')
    st.write(' ')
    
    # Input text area for user input (allowing multiline input)
    user_input = st.text_area('Enter text to classify (one text per line or separated by commas):', height=200)

    # Button to load sample text
    if st.button('Load Sample Text'):
        sample_texts = [
            "Jet Airways, India's biggest airline, had a successful IPO selling out in 10 minutes. The company plans to use the funds to buy new planes and reduce debt.",
            "\nTony Blair emphasized prosperity for all in his speech, aiming to quell rumors of disagreements with Gordon Brown. Labour is focusing on low inflation and improved public services for the upcoming election.",
            "\nClint Eastwood's 'Million Dollar Baby' won Best Picture at the Oscars, beating out Martin Scorsese's 'The Aviator.' Hilary Swank and Morgan Freeman won acting awards for the film.",
            "\nA new file-sharing network called Exeem is being developed to avoid legal shutdowns experienced by previous networks like BitTorrent. It uses a combination of BitTorrent and Kazaa's systems.",
            "\nBBC's online search engine saw a record number of inquiries in 2004, including both common and strange requests. Some of the oddest questions included how to fold a napkin like an elf's boot and what the biggest collection of naval fluff is."
        ]
        
        # Join sample texts into one string separated by '\n'
        sample_text = '\n'.join(sample_texts)
        st.text_area('Sample Text:', value=sample_text, height=200)

    if st.button('Classify'):
        if user_input.strip() == '':
            st.warning('Please enter some text.')
        else:
            # Split the user input into individual texts
            texts = [text.strip() for text in user_input.split('\n') if text.strip()]
            
            # Perform prediction for each text
            predictions = []
            for text in texts:
                prediction = predict(text)
                predicted_label = prediction_mapping.get(prediction, 'Unknown')
                predictions.append({'Text': text, 'Prediction Label': predicted_label})
            
            # Display predictions in a DataFrame
            prediction_df = pd.DataFrame(predictions)
            st.success('Predictions:')
            st.dataframe(prediction_df)
            
            
if __name__ == '__main__':
    main()
