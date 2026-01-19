

## **10. Web Interface with Gradio (10 Marks)**

#Create a user-friendly Gradio web interface that takes user inputs and displays the prediction from your trained model.


import gradio as gr
import pandas as pd
import pickle



with open("insurance_model.pkl", "rb") as f:
    pipeline = pickle.load(f)


# Prediction function
def predict_cost(age, sex, bmi, children, smoker, region):

    # Create BMI category
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region],
        "bmi_category": [bmi_category]
    })
    return pipeline.predict(input_df)[0]


demo = gr.Interface(
    fn=predict_cost,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(["male","female"], label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Children"),
        gr.Dropdown(["yes","no"], label="Smoker"),
        gr.Dropdown(["northeast","northwest","southeast","southwest"], label="Region")
    ],
    outputs="number",
    title="Medical Insurance Cost Predictor",
    description="Enter your details to predict your medical insurance cost."
)

demo.launch(share=True)

"""## **11. Deployment to Hugging Face (10 Marks)**

Hugging Face Spaces public URL:

"""