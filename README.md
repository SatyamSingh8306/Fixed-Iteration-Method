# Math Solution Assistant with Groq AI

This is a Streamlit application that leverages LangChain and Groq's LLMs to create an interactive assistant for mathematical solutions. It's especially focused on fixed-point iteration methods, but can handle a wide range of math and programming queries.

## Features

- **Chat with Groq LLMs**: Uses the Llama3-70B model for intelligent conversations
- **Document Processing**: Upload PDFs, text files, or code files to provide context
- **Voice Interactions**: Upload audio files for speech-to-text processing
- **Text-to-Speech**: Listen to AI responses with gTTS
- **Image Upload**: Upload images for context
- **Fixed Point Iteration Solver**: Special agent for solving mathematical equations
- **Vector Database**: Creates embeddings from documents for context-aware responses

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Get a Groq API key from [groq.com](https://groq.com)
4. Run the app:
   ```
   streamlit run app.py
   ```

## Usage

1. Enter your Groq API key in the sidebar
2. Optional: Upload a document for context (PDF, TXT, code files)
3. Initialize the conversation (with or without document context)
4. Ask questions or upload images/audio for processing
5. For fixed-point iteration problems, use the Fixed Point Solver Agent

## Example Math Queries

- "Solve the equation cos(x) - x = 0 using fixed point iteration"
- "Explain the convergence criteria for fixed point iteration"
- "Help me understand the code for analyzing convergence"
- "What does the error plot tell us about the solution?"

## Requirements

Python 3.8+ and the packages listed in requirements.txt

## License

MIT