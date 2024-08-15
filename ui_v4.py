import streamlit as st
from datetime import datetime
import random
import numpy as np
import pandas as pd
import os
import base64
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import classes_main_v2
from ics import Calendar, Event


# Example greeting responses
greetings = ["hello", "hi", "hey", "greetings", "what's up", "howdy"]
greeting_responses = ["Hello! How can I assist you today?", "Hi there! What brings you here?", "Greetings! How can I help you?"]

# Initialize the Qwen2 0.5B model
def initialize_language_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    return model, tokenizer

model, tokenizer = initialize_language_model()

# Function to generate response using Qwen2 0.5B
def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    messages = [
        {"role": "system", "content": """You are a helpful medical assistant.
        Provide accurate and helpful information, but always recommend
        consulting with a healthcare professional for personalized medical advice.

        Remember -  your results will always be in ENGLISH"""},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=400, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response because than we can split further
    assistant_response = response.split("assistant")[-1].strip()

    # Remove any remaining system or user prompts for final cleaner respsonse
    clean_response = assistant_response.split("system")[-1].split("user")[-1].strip()

    return clean_response

# Function to create an ICS file for the appointment
def create_ics_file(name, email, appointment_time):
    calendar = Calendar()
    event = Event()
    event.name = f"Appointment with {name}"
    event.begin = appointment_time.strftime("%Y-%m-%d %H:%M:%S")
    event.description = f"Appointment booked by {name}. Contact: {email}"
    calendar.events.add(event)
    return str(calendar)

# Function to generate a download link for the ICS file
def get_binary_file_downloader_html(bin_file, file_name):
    b64 = base64.b64encode(bin_file).decode()
    href = f'<a href="data:file/ics;base64,{b64}" download="{file_name}">Download Appointment</a>'
    return href

# Function to run the chatbot with Streamlit UI
def run_chatbot_ui(classifier):
    st.title("AI Medical Chatbot")
    st.write("Welcome to the AI Medical Chatbot. How are you feeling today?")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_booking_form" not in st.session_state:
        st.session_state.show_booking_form = False

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What are your symptoms?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        if prompt.lower() in ['quit', 'exit', 'bye', 'thanks', 'it was helpful', 'thankyou']:
            response = "Goodbye! Take care of your health."
        elif prompt.lower() in greetings:
            response = random.choice(greeting_responses)
        else:
            entities = classes_main_v2.extract_medical_entities(prompt)
            processed_input = classes_main_v2.text_preprocessing(prompt)
            sbert_features = classes_main_v2.extract_sbert_features([processed_input])

            predicted_label = classifier.predict(sbert_features)[0]
            predicted_proba = classifier.predict_proba(sbert_features)[0]
            top_3_labels = classifier.classes_[np.argsort(predicted_proba)[-1:][::-1]]
            top_3_proba = predicted_proba[np.argsort(predicted_proba)[-1:][::-1]]

            ai_prompt = f"Patient input: {prompt}\n"
            if any(entities.values()):
                ai_prompt += f"Relevant medical entities: {entities}\n"
            ai_prompt += "Possible conditions (with probabilities): \n"
            for label, prob in zip(top_3_labels, top_3_proba):
                ai_prompt += f"- {label}: {prob:.2f}\n"
            ai_prompt += "Provide a helpful and empathetic response, considering the possible conditions. Ask for more information if needed, and always recommend consulting a healthcare professional."

            response = generate_response(model, tokenizer, ai_prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)

    # Booking button
    if st.button("Book an Appointment"):
        st.session_state.show_booking_form = True

    # Booking form
    if st.session_state.show_booking_form:
        with st.form("appointment_form"):
            st.write("Please enter your details to book an appointment:")
            name = st.text_input("Name")
            email = st.text_input("Email")
            date = st.date_input("Preferred Date")
            time_input = st.time_input("Preferred Time")
            
            submitted = st.form_submit_button("Confirm Booking")
            if submitted:
                if name and email and date and time_input:
                    appointment_time = datetime.combine(date, time_input)
                    ics_content = create_ics_file(name, email, appointment_time)
                    st.success("Appointment booked successfully!")
                    st.markdown(get_binary_file_downloader_html(ics_content.encode(), 'appointment.ics'), unsafe_allow_html=True)
                    
                    # Notify user and start a 30-second sleep
                    st.success("You will be redirected in 30 seconds...")
                    time.sleep(30)  # Wait for 30 seconds
                    st.session_state.messages = []
                    st.session_state.show_booking_form = False
                    st.experimental_rerun()  # Rerun the app to reflect changes
                else:
                    st.error("Please fill in all the fields.")

    # Reset button
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.show_booking_form = False
        st.experimental_rerun()

# Main execution
def main():
    # Load and preprocess data
    data = classes_main_v2.load_and_preprocess_data('New_Symptoms_Biggest.csv')

    # Train the classifier
    classifier = classes_main_v2.train_classifier(data)

    # Run the chatbot with Streamlit UI
    run_chatbot_ui(classifier)

if __name__ == "__main__":
    main()
