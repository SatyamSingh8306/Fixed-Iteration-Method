#!/bin/bash

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Set environment variables
echo "Setting up environment variables..."
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

echo "Setup complete! Run the application with: streamlit run app.py"
echo "Remember to update your Groq API key in the .env file"