# Multiclass Text Classification Fine Tuning with DistilBERT using Streamlit

## Purpose:
The Python script uses Streamlit along with a pre-trained DistilBERT model to classify text input into five categories: business, entertainment, politics, sport, and tech.

## Libraries Used:
- `streamlit`: For creating the interactive web application.
- `transformers.DistilBertTokenizer`: Tokenizer for tokenizing input text.
- `transformers.TFDistilBertForSequenceClassification`: Pre-trained DistilBERT model for sequence classification.
- `tensorflow`: TensorFlow framework for model operations.
- `pandas`: For handling data structures.

## Components of the Script:

1. **Loading the Model and Tokenizer**:
   - The script loads a pre-trained DistilBERT model (`TFDistilBertForSequenceClassification`) and its tokenizer (`DistilBertTokenizer`) from the specified directory (`model_path`).

2. **Prediction Mapping**:
   - `prediction_mapping` defines a mapping from numerical predictions to human-readable categories.

3. **Prediction Function (`predict`)**:
   - `predict(input_text)`: Tokenizes the input text using the tokenizer, passes it through the loaded model, and predicts the category using `argmax` on the logits.

4. **Streamlit Application (`main`)**:
   - Sets up the Streamlit application with a title and description of the model and classification categories.
   - Provides a text area for users to input text for classification.
   - Includes a button to load sample text and display it in the text area.
   - Executes classification upon clicking the 'Classify' button:
     - Splits the input text into individual texts.
     - Calls `predict` function for each text to get predictions.
     - Displays the results in a pandas DataFrame within the Streamlit interface.

5. **User Interface Elements**:
   - **Input Text Area**: Allows users to enter text for classification.
   - **Load Sample Text Button**: Loads predefined sample texts into the input text area.
   - **Classify Button**: Triggers the classification process and displays the results in a structured format.

## Usage:
To use the script:
- Ensure Streamlit and required Python libraries (`transformers`, `tensorflow`, `pandas`) are installed.
- Set `model_path` to the directory containing the TensorFlow weights for the DistilBERT model.
- Run the script using `streamlit run script_name.py` in the terminal.
- Interact with the web interface to classify text into one of the predefined categories.

This setup enables easy deployment and interaction with a text classification model using a web-based interface powered by Streamlit.


## 🔗 Connect with Me

Feel free to connect with me on LinkedIn:

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://parthebhan143.wixsite.com/datainsights)

[![LinkedIn Profile](https://img.shields.io/badge/LinkedIn_Profile-000?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/parthebhan)

[![Kaggle Profile](https://img.shields.io/badge/Kaggle_Profile-000?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/parthebhan)

[![Tableau Profile](https://img.shields.io/badge/Tableau_Profile-000?style=for-the-badge&logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/parthebhan.pari/vizzes)

