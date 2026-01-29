import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('titanic_voting_model.pkl')

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create a DataFrame from the inputs
    data = pd.DataFrame({
        'Pclass': [int(pclass)],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [int(sibsp)],
        'Parch': [int(parch)],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    
    # Predict
    prediction = model.predict(data)
    
    if prediction[0] == 1:
        return "Survived"
    else:
        return "Did not survive"

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Radio([1, 2, 3], label="Pclass (Ticket Class)"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Slider(0, 100, value=30, label="Age"),
        gr.Slider(0, 10, value=0, step=1, label="SibSp (Siblings/Spouses aboard)"),
        gr.Slider(0, 10, value=0, step=1, label="Parch (Parents/Children aboard)"),
        gr.Number(value=32.0, label="Fare"),
        gr.Radio(["S", "C", "Q"], label="Embarked (Port of Embarkation)")
    ],
    outputs="text",
    title="Titanic Survivor Prediction",
    description="Enter passenger details to predict survival."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
