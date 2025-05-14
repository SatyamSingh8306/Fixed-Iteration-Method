# import streamlit as st
# import os
# import tempfile
# import base64
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# from io import BytesIO
# import speech_recognition as sr
# from gtts import gTTS
# from PIL import Image
# import io
# import time
# import sys

# # LangChain & Groq imports
# from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain.agents import initialize_agent, Tool
# from langchain.agents import AgentType
# from langchain.schema import HumanMessage, AIMessage

# # Import the fixed point iteration module (assuming it's in the same directory)
# from fixed_point_iteration_module import run_fixed_point_iteration, fixed_point_iteration, create_function_from_expression, parse_and_rearrange_equation

# # Set page configuration
# st.set_page_config(page_title="Math Solution Assistant", layout="wide")

# # Initialize session state variables
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "conversation" not in st.session_state:
#     st.session_state.conversation = None
# if "groq_api_key" not in st.session_state:
#     st.session_state.groq_api_key = ""
# if "file_content" not in st.session_state:
#     st.session_state.file_content = ""
# if "audio_path" not in st.session_state:
#     st.session_state.audio_path = None

# # Define functions for handling files
# def process_uploaded_file(uploaded_file):
#     file_extension = uploaded_file.name.split(".")[-1].lower()
#     content = ""
    
#     if file_extension == "pdf":
#         with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
#             temp_file.write(uploaded_file.read())
#             temp_path = temp_file.name
        
#         loader = PyPDFLoader(temp_path)
#         documents = loader.load()
#         os.unlink(temp_path)
        
#         for doc in documents:
#             content += doc.page_content + "\n\n"
            
#     elif file_extension in ["txt", "py", "java", "cpp", "html", "css", "js"]:
#         content = uploaded_file.getvalue().decode("utf-8")
    
#     else:
#         st.warning(f"File format '{file_extension}' is not supported for text extraction.")
        
#     return content

# def handle_image_upload(uploaded_file):
#     if uploaded_file is not None:
#         # Read image file
#         image = Image.open(uploaded_file)
        
#         # Display the image
#         st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        
#         # Return the image for further processing
#         return image
#     return None

# def handle_audio_upload(uploaded_file):
#     if uploaded_file is not None:
#         # Save audio file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
#             temp_file.write(uploaded_file.read())
#             temp_path = temp_file.name
        
#         st.audio(uploaded_file, format="audio/wav")
        
#         # Convert speech to text
#         r = sr.Recognizer()
#         with sr.AudioFile(temp_path) as source:
#             audio_data = r.record(source)
#             try:
#                 text = r.recognize_google(audio_data)
#                 st.success(f"Transcription: {text}")
#                 return text
#             except sr.UnknownValueError:
#                 st.error("Speech Recognition could not understand the audio")
#             except sr.RequestError as e:
#                 st.error(f"Could not request results from Speech Recognition service; {e}")
        
#         # Clean up temp file
#         os.unlink(temp_path)
    
#     return None

# def text_to_speech(text):
#     tts = gTTS(text=text, lang='en')
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
#         temp_path = temp_file.name
    
#     tts.save(temp_path)
#     st.session_state.audio_path = temp_path
    
#     audio_bytes = open(temp_path, "rb").read()
#     audio_b64 = base64.b64encode(audio_bytes).decode()
    
#     st.markdown(
#         f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
#         unsafe_allow_html=True
#     )
    
#     return temp_path

# def create_vectors_from_text(text):
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = text_splitter.split_text(text)
    
#     # Use HuggingFace embeddings (offline option)
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
#     # Create vector store
#     vector_store = FAISS.from_texts(chunks, embeddings)
#     return vector_store

# def initialize_conversation_chain(api_key, vector_store=None):
#     # Initialize the ChatGroq LLM
#     llm = ChatGroq(
#         api_key=api_key,
#         model_name="llama3-70b-8192",
#         temperature=0.5,
#     )
    
#     # Set up memory
#     memory = ConversationBufferMemory(
#         memory_key="chat_history", 
#         return_messages=True,
#         output_key="answer"
#     )

#     # Define system prompt    
#     system_template = """
#     You are a helpful math and programming assistant that helps users understand and work with mathematical
#     concepts, algorithms, and code. You have special expertise in fixed-point iteration methods and numerical
#     analysis techniques.

#     When responding to queries:
#     1. If the question involves mathematical concepts, provide explanations with clear examples
#     2. For code-related questions, explain the code functionality and how to improve or debug it
#     3. For fixed-point iteration specifically, explain the method, convergence criteria, and practical applications
#     4. Use conversational language and be pedagogically helpful

#     Context information from documents: {context}
    
#     Current conversation: {chat_history}
    
#     Human: {question}
#     AI:
#     """
    
#     prompt = PromptTemplate(
#         input_variables=["context", "chat_history", "question"],
#         output_key="answer",
#         template=system_template
#     )
    
#     # Create the chain based on whether we have a vector store or not
#     if vector_store:
#         conversation_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
#             memory=memory,
#             combine_docs_chain_kwargs={"prompt": prompt},
#             return_source_documents=True,
#             verbose=True
#         )
#     else:
#         # Simple conversation chain without retrieval
#         from langchain.chains import LLMChain
#         conversation_chain = LLMChain(
#             llm=llm,
#             prompt=prompt,
#             memory=memory,
#             verbose=True
#         )
    
#     return conversation_chain

# def create_fixed_point_agent(api_key):
#     # Initialize the ChatGroq LLM
#     llm = ChatGroq(
#         api_key=api_key,
#         model_name="llama3-70b-8192",
#         temperature=0.5,
#     )
    
#     # Create tools for the agent
#     def solve_equation(equation_str):
#         """
#         Solve an equation using fixed point iteration method.
#         Input should be an equation like 'cos(x) - x = 0'.
#         """
#         try:
#             # Parse and rearrange
#             rhs_expr, f_expr = parse_and_rearrange_equation(equation_str)
#             if rhs_expr is None:
#                 return "Failed to parse the equation. Please check the syntax."
            
#             # Create functions
#             g = create_function_from_expression(rhs_expr)
#             f = create_function_from_expression(f_expr)
            
#             # Run fixed point iteration with default values
#             x0 = 0.5  # Default initial guess
#             tol = 1e-6
#             max_iter = 100
            
#             x_values, iterations, converged = fixed_point_iteration(g, x0, tol, max_iter)
            
#             # Format results
#             result = f"Equation: {f_expr} = 0\n"
#             result += f"Fixed-point form: x = {rhs_expr}\n"
#             result += f"Initial guess: x0 = {x0}\n"
#             result += f"Final approximation: x = {x_values[-1]}\n"
#             result += f"Function value at solution: f(x) = {f(x_values[-1])}\n"
#             result += f"Iterations: {iterations}\n"
#             result += f"Converged: {converged}\n"
            
#             # Create convergence plot
#             plt.figure(figsize=(8, 6))
#             plt.plot(range(len(x_values)), x_values, 'bo-', label='Approximations')
#             plt.grid(True)
#             plt.xlabel('Iteration')
#             plt.ylabel('Approximation')
#             plt.title('Convergence of Fixed Point Iteration Method')
#             plt.legend()
            
#             # Create error plot if converged
#             if converged and len(x_values) > 1:
#                 plt.figure(figsize=(8, 6))
#                 errors = [abs(x_values[i] - x_values[i-1]) for i in range(1, len(x_values))]
#                 plt.semilogy(range(1, len(x_values)), errors, 'ro-', label='Error')
#                 plt.grid(True)
#                 plt.xlabel('Iteration')
#                 plt.ylabel('Error (log scale)')
#                 plt.title('Error Convergence - Fixed Point Iteration')
#                 plt.legend()
            
#             # Display iteration table
#             result += "\n\nIteration Table:\n"
#             result += "Iteration | Approximation | |x_n - x_{n-1}| | f(x_n)\n"
#             result += "-" * 60 + "\n"
            
#             for i, x in enumerate(x_values):
#                 if i == 0:
#                     result += f"{i:9d} | {x:.10f} | {'N/A':14} | {f(x):.10e}\n"
#                 else:
#                     error = abs(x - x_values[i-1])
#                     result += f"{i:9d} | {x:.10f} | {error:.10e} | {f(x):.10e}\n"
            
#             # Add convergence analysis
#             if converged:
#                 try:
#                     derivative = (g(x_values[-1] + 0.01) - g(x_values[-1] - 0.01)) / (2 * 0.01)
#                     abs_derivative = abs(derivative)
                    
#                     result += f"\nConvergence analysis: |g'(x*)| ≈ {abs_derivative:.6f}\n"
                    
#                     if abs_derivative < 1:
#                         result += "Convergence status: Convergent (|g'(x*)| < 1)"
#                     elif abs_derivative > 1:
#                         result += "Convergence status: Divergent (|g'(x*)| > 1)"
#                     else:
#                         result += "Convergence status: Indeterminate (|g'(x*)| = 1)"
#                 except:
#                     pass
            
#             return result
            
#         except Exception as e:
#             return f"Error solving equation: {str(e)}"
    
#     # Create tools list
#     tools = [
#         Tool(
#             name="FixedPointSolver",
#             func=solve_equation,
#             description="Solves mathematical equations using fixed point iteration method. Input should be an equation like 'cos(x) - x = 0'."
#         )
#     ]
    
#     # Initialize memory for the agent
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
#     # Initialize agent with memory
#     agent = initialize_agent(
#         tools,
#         llm,
#         agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#         verbose=True,
#         max_iterations=3,
#         memory=memory,
#         handle_parsing_errors=True
#     )
    
#     return agent

# # Sidebar for API key and file upload
# with st.sidebar:
#     st.title("Configuration")
    
#     # API Key input
#     api_key = st.text_input("Enter your Groq API Key:", value=st.session_state.groq_api_key, type="password")
#     api_key = os.environ.get("GROQ_API_KEY")
#     if api_key:
#         st.session_state.groq_api_key = api_key
    
#     st.divider()
#     st.subheader("Upload Files")
    
#     # Text file upload
#     uploaded_file = st.file_uploader("Upload a document (PDF, TXT, or code files):", type=["pdf", "txt", "py", "java", "cpp", "html", "css", "js"])
#     if uploaded_file is not None:
#         content = process_uploaded_file(uploaded_file)
#         st.session_state.file_content = content
#         st.success(f"Successfully processed {uploaded_file.name}")
        
#         if st.button("Initialize Conversation with Document"):
#             with st.spinner("Creating vector database from document..."):
#                 vector_store = create_vectors_from_text(content)
#                 st.session_state.conversation = initialize_conversation_chain(st.session_state.groq_api_key, vector_store)
#             st.success("Conversation initialized with document content!")
    
#     st.divider()
    
#     # Image upload
#     st.subheader("Upload Image")
#     uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
#     if uploaded_image is not None:
#         image = handle_image_upload(uploaded_image)
    
#     st.divider()
    
#     # Audio upload
#     st.subheader("Audio Input")
#     uploaded_audio = st.file_uploader("Upload an audio file:", type=["wav"])
#     if uploaded_audio is not None:
#         transcription = handle_audio_upload(uploaded_audio)
#         if transcription and st.button("Send Transcription"):
#             st.session_state.messages.append({"role": "user", "content": transcription})
    
#     # Audio recording option
#     if st.button("Record Audio"):
#         with st.spinner("Recording audio for 5 seconds..."):
#             # Placeholder for audio recording (would require JavaScript integration)
#             st.info("Audio recording feature requires JavaScript integration. Use the file upload instead.")
    
#     st.divider()
    
#     # Initialize conversation without document
#     if st.button("Initialize Simple Conversation (No Document)"):
#         if st.session_state.groq_api_key:
#             st.session_state.conversation = initialize_conversation_chain(st.session_state.groq_api_key)
#             st.success("Simple conversation initialized!")
#         else:
#             st.error("Please enter your Groq API key first!")
    
#     # Initialize fixed point agent
#     if st.button("Initialize Fixed Point Solver Agent"):
#         if st.session_state.groq_api_key:
#             st.session_state.conversation = create_fixed_point_agent(st.session_state.groq_api_key)
#             st.success("Fixed Point Solver Agent initialized!")
#         else:
#             st.error("Please enter your Groq API key first!")

# # Main page
# st.title("Math Solution Assistant with Groq AI")

# # Display conversation
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

#     # Chat input
# if prompt := st.chat_input("Ask a question about mathematics, algorithms, or upload content"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Generate and display assistant response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             if st.session_state.conversation is not None:
#                 if isinstance(st.session_state.conversation, ConversationalRetrievalChain):
#                     response = st.session_state.conversation({"question": prompt})
#                     ai_response = response["answer"]
#                 else:
#                     # Handle agent or simple LLMChain
#                     try:
#                         if hasattr(st.session_state.conversation, 'run'):
#                             # For regular LLMChain
#                             response = st.session_state.conversation.run(input=prompt)
#                             ai_response = response
#                         else:
#                             # For agent expecting chat_history
#                             chat_history = []
#                             for msg in st.session_state.messages:
#                                 if msg["role"] == "user":
#                                     chat_history.append(HumanMessage(content=msg["content"]))
#                                 elif msg["role"] == "assistant":
#                                     chat_history.append(AIMessage(content=msg["content"]))
                            
#                             # Agent call with chat_history
#                             response = st.session_state.conversation({"input": prompt, "chat_history": chat_history})
#                             ai_response = response["output"] if "output" in response else response
#                     except Exception as e:
#                         ai_response = f"Error processing your request: {str(e)}"
#             else:
#                 ai_response = "Please initialize the conversation first using the sidebar options."
        
#         st.markdown(ai_response)
        
#         # Text-to-speech option
#         if st.button("Listen to Response"):
#             with st.spinner("Generating audio..."):
#                 audio_path = text_to_speech(ai_response)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": ai_response})

# # Clean up temporary audio files when the app is closed
# def cleanup():
#     if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
#         os.unlink(st.session_state.audio_path)

# # Register cleanup function to run when app shuts down
# import atexit
# atexit.register(cleanup)

import streamlit as st
import os
import tempfile
import base64
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import speech_recognition as sr
from gtts import gTTS
from PIL import Image
import io
import time
import sys
import pandas as pd

# LangChain & Groq imports
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.schema import HumanMessage, AIMessage

# Import the fixed point iteration module (assuming it's in the same directory)
from fixed_point_iteration_module import run_fixed_point_iteration, fixed_point_iteration, create_function_from_expression, parse_and_rearrange_equation, plot_convergence

# Set page configuration
st.set_page_config(page_title="Math Solution Assistant", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "equation_results" not in st.session_state:
    st.session_state.equation_results = None
if "chat_plots" not in st.session_state:
    st.session_state.chat_plots = {}
if "solution_steps" not in st.session_state:
    st.session_state.solution_steps = {}

# Define functions for handling files
def process_uploaded_file(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    content = ""
    
    if file_extension == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        os.unlink(temp_path)
        
        for doc in documents:
            content += doc.page_content + "\n\n"
            
    elif file_extension in ["txt", "py", "java", "cpp", "html", "css", "js"]:
        content = uploaded_file.getvalue().decode("utf-8")
    
    else:
        st.warning(f"File format '{file_extension}' is not supported for text extraction.")
        
    return content

def handle_image_upload(uploaded_file):
    if uploaded_file is not None:
        # Read image file
        image = Image.open(uploaded_file)
        
        # Display the image
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        
        # Return the image for further processing
        return image
    return None

def handle_audio_upload(uploaded_file):
    if uploaded_file is not None:
        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        
        st.audio(uploaded_file, format="audio/wav")
        
        # Convert speech to text
        r = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data)
                st.success(f"Transcription: {text}")
                return text
            except sr.UnknownValueError:
                st.error("Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                st.error(f"Could not request results from Speech Recognition service; {e}")
        
        # Clean up temp file
        os.unlink(temp_path)
    
    return None

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_path = temp_file.name
    
    tts.save(temp_path)
    st.session_state.audio_path = temp_path
    
    audio_bytes = open(temp_path, "rb").read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    
    st.markdown(
        f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True
    )
    
    return temp_path

def create_vectors_from_text(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    # Use HuggingFace embeddings (offline option)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def initialize_conversation_chain(api_key, vector_store=None):
    # Initialize the ChatGroq LLM
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama3-70b-8192",
        temperature=0.5,
    )
    
    # Set up memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )

    # Define system prompt    
    system_template = """
    You are a helpful math and programming assistant that helps users understand and work with mathematical
    concepts, algorithms, and code. You have special expertise in fixed-point iteration methods and numerical
    analysis techniques.

    When responding to queries:
    1. If the question involves mathematical concepts, provide explanations with clear examples
    2. For code-related questions, explain the code functionality and how to improve or debug it
    3. For fixed-point iteration specifically, explain the method, convergence criteria, and practical applications
    4. Use conversational language and be pedagogically helpful
    5. Identify if the query seems to be asking for a fixed-point iteration solution, but don't solve it yourself - the app will handle that

    Context information from documents: {context}
    
    Current conversation: {chat_history}
    
    Human: {question}
    AI:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        output_key="answer",
        template=system_template
    )
    
    # Create the chain based on whether we have a vector store or not
    if vector_store:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
    else:
        # Simple conversation chain without retrieval
        from langchain.chains import LLMChain
        conversation_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
    
    return conversation_chain

def analyze_convergence(g, x_star, delta=0.01):
    """
    Analyze convergence based on derivative magnitude
    """
    # Approximate the derivative using central difference
    derivative = (g(x_star + delta) - g(x_star - delta)) / (2 * delta)
    abs_derivative = abs(derivative)
    
    if abs_derivative < 1:
        return derivative, "Convergent (|g'(x*)| < 1)"
    elif abs_derivative > 1:
        return derivative, "Divergent (|g'(x*)| > 1)"
    else:
        return derivative, "Indeterminate (|g'(x*)| = 1)"

def solve_fixed_point_equation(equation_str, x0=0.5, tol=1e-6, max_iter=100):
    """
    Dedicated function to solve and display fixed point iteration results directly in the app
    """
    try:
        # Parse and rearrange the equation
        rhs_expr, f_expr = parse_and_rearrange_equation(equation_str)
        if rhs_expr is None:
            return {"error": "Failed to parse the equation. Please check the syntax."}
        
        # Create the functions
        g = create_function_from_expression(rhs_expr)
        f = create_function_from_expression(f_expr)
        
        # Run the fixed point iteration
        x_values, iterations, converged = fixed_point_iteration(g, x0, tol, max_iter)
        
        # Create result dictionary
        result = {
            "original_equation": f"{f_expr} = 0",
            "fixed_point_form": f"x = {rhs_expr}",
            "initial_guess": x0,
            "final_approximation": x_values[-1],
            "function_value": f(x_values[-1]),
            "iterations": iterations,
            "converged": converged,
            "x_values": x_values,
            "iteration_errors": [abs(x_values[i] - x_values[i-1]) for i in range(1, len(x_values))],
            "function_values": [f(x) for x in x_values]
        }
        
        # Analyze convergence if method converged
        if converged:
            derivative, convergence_status = analyze_convergence(g, x_values[-1])
            result["derivative_abs"] = abs(derivative)
            result["convergence_status"] = convergence_status
        
        return result
    
    except Exception as e:
        return {"error": f"Error solving equation: {str(e)}"}

def create_iteration_plots(results, message_idx):
    """
    Create plots for the fixed point iteration results and store them in session state
    """
    plots = {}
    
    # Create convergence plot
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(len(results['x_values'])), results['x_values'], 'bo-', label='Approximations')
    ax1.axhline(y=results['final_approximation'], color='r', linestyle='--', 
               label=f"Solution: {results['final_approximation']:.6f}")
    ax1.grid(True)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Approximation')
    ax1.set_title('Convergence of Fixed Point Iteration')
    ax1.legend()
    
    # Save plot to BytesIO
    buf1 = BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    plots['convergence'] = buf1
    
    # Create error plot if there are iterations
    if len(results['iteration_errors']) > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.semilogy(range(1, len(results['x_values'])), results['iteration_errors'], 'ro-', label='Error')
        ax2.grid(True)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Error (log scale)')
        ax2.set_title('Error Convergence')
        ax2.legend()
        
        # Save plot to BytesIO
        buf2 = BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)
        plots['error'] = buf2
    
    # Store plots in session state
    st.session_state.chat_plots[message_idx] = plots
    
    return plots

def create_solution_steps(results, message_idx):
    """
    Create detailed solution steps for the fixed point iteration
    """
    steps = []
    
    # Step 1: Original equation
    steps.append({
        "title": "Step 1: Original Equation",
        "content": f"The original equation is:\n\n**{results['original_equation']}**"
    })
    
    # Step 2: Rearranged form
    steps.append({
        "title": "Step 2: Rearranged Form",
        "content": f"The equation is rearranged to fixed-point form:\n\n**{results['fixed_point_form']}**"
    })
    
    # Step 3: Initialization
    steps.append({
        "title": "Step 3: Initialization",
        "content": f"Starting with initial guess:\n\n**x₀ = {results['initial_guess']}**"
    })
    
    # Step 4: Iteration process
    iteration_details = "The iteration process follows:\n\n"
    iteration_details += "| Iteration | Approximation | Error | f(x) |\n"
    iteration_details += "|-----------|---------------|-------|------|\n"
    
    for i, x in enumerate(results['x_values']):
        if i == 0:
            iteration_details += f"| {i} | {x:.6f} | N/A | {results['function_values'][i]:.6e} |\n"
        else:
            error = results['iteration_errors'][i-1]
            iteration_details += f"| {i} | {x:.6f} | {error:.6e} | {results['function_values'][i]:.6e} |\n"
    
    steps.append({
        "title": "Step 4: Iteration Process",
        "content": iteration_details
    })
    
    # Step 5: Final result
    steps.append({
        "title": "Step 5: Final Result",
        "content": f"The method {'converged' if results['converged'] else 'did not converge'} after {results['iterations']} iterations.\n\n"
                  f"**Final approximation:** {results['final_approximation']:.10f}\n"
                  f"**Function value at solution:** {results['function_value']:.10e}"
    })
    
    # Step 6: Convergence analysis (if converged)
    if results['converged'] and "derivative_abs" in results:
        steps.append({
            "title": "Step 6: Convergence Analysis",
            "content": f"The derivative at the solution point is approximately |g'(x*)| ≈ {results['derivative_abs']:.6f}\n\n"
                      f"**{results['convergence_status']}**"
        })
    
    # Store steps in session state
    st.session_state.solution_steps[message_idx] = steps
    
    return steps

def detect_equation_query(prompt):
    """
    Detect if the prompt is asking to solve an equation using fixed point iteration
    Returns (is_equation_query, equation_str) if detected, else (False, None)
    """
    # Simple keyword detection (can be improved with regex or NLP)
    keywords = ["solve", "equation", "fixed point", "fixed-point", "iteration", "find root", "find the root"]
    equation_indicators = ["=", "sin", "cos", "tan", "exp", "log", "ln", "sqrt", "^", "**"]
    
    has_keywords = any(keyword.lower() in prompt.lower() for keyword in keywords)
    has_equation = any(indicator in prompt for indicator in equation_indicators)
    
    if has_keywords and has_equation:
        # Extract the equation (simple heuristic, can be improved)
        lines = prompt.split('\n')
        for line in lines:
            if '=' in line:
                return True, line.strip()
        
        # If no equation with = is found, just return the prompt
        return True, prompt
    
    return False, None

def extract_parameters_from_prompt(prompt):
    """
    Extract fixed point iteration parameters from the prompt if specified
    """
    x0 = 0.5  # default
    tol = 1e-6  # default
    max_iter = 100  # default
    
    # Check for initial guess specification
    if "initial" in prompt.lower() and "guess" in prompt.lower():
        try:
            # Look for patterns like "initial guess = 0.7" or "x0 = 0.7"
            x0_patterns = ["initial guess = ", "initial guess:", "x0 = ", "x0:", "x_0 = ", "x_0:"]
            for pattern in x0_patterns:
                if pattern in prompt.lower():
                    parts = prompt.lower().split(pattern)[1].strip().split()
                    if parts and parts[0].replace('.', '', 1).isdigit():
                        x0 = float(parts[0])
                        break
        except:
            pass
    
    # Check for tolerance specification
    if "tolerance" in prompt.lower() or "tol" in prompt.lower():
        try:
            tol_patterns = ["tolerance = ", "tolerance:", "tol = ", "tol:"]
            for pattern in tol_patterns:
                if pattern in prompt.lower():
                    parts = prompt.lower().split(pattern)[1].strip().split()
                    if parts and parts[0].replace('.', '', 1).replace('e-', '', 1).isdigit():
                        tol = float(parts[0])
                        break
        except:
            pass
    
    # Check for max iterations
    if "max" in prompt.lower() and ("iter" in prompt.lower() or "iteration" in prompt.lower()):
        try:
            iter_patterns = ["max iter = ", "max iter:", "max iterations = ", "max iterations:", "iterations = ", "iterations:"]
            for pattern in iter_patterns:
                if pattern in prompt.lower():
                    parts = prompt.lower().split(pattern)[1].strip().split()
                    if parts and parts[0].isdigit():
                        max_iter = int(parts[0])
                        break
        except:
            pass
    
    return x0, tol, max_iter

# Add a standalone fixed point solver UI for the tab interface
def fixed_point_solver_ui():
    st.header("Fixed Point Iteration Method Solver")
    st.markdown("""
    This solver finds the root of a function using the fixed point iteration method.
    Enter an equation in the form `f(x) = 0` or `y = f(x)`.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        equation = st.text_input("Enter the equation to solve:", 
                              placeholder="e.g., cos(x) - 3*x + 1 = 0 or y = cos(x) - 3*x + 1")
        
        x0 = st.number_input("Initial guess (x₀):", value=0.5, step=0.1)
        tol = st.number_input("Tolerance:", value=1e-6, format="%e", step=1e-6)
        max_iter = st.number_input("Maximum iterations:", value=100, step=10)
    
    if st.button("Solve Equation"):
        if equation:
            with st.spinner("Solving equation..."):
                results = solve_fixed_point_equation(equation, x0, tol, max_iter)
                st.session_state.equation_results = results
        else:
            st.error("Please enter an equation to solve.")
    
    # Display results if available
    if st.session_state.equation_results:
        results = st.session_state.equation_results
        
        if "error" in results:
            st.error(results["error"])
        else:
            with col2:
                st.subheader("Solution Results")
                st.markdown(f"""
                **Original equation:** {results['original_equation']}  
                **Fixed-point form:** {results['fixed_point_form']}  
                **Initial guess:** {results['initial_guess']}  
                **Final approximation:** {results['final_approximation']:.10f}  
                **Function value at solution:** {results['function_value']:.10e}  
                **Iterations required:** {results['iterations']}  
                **Converged:** {results['converged']}
                """)
                
                if results['converged'] and "derivative_abs" in results:
                    st.markdown(f"""
                    **Convergence analysis:**  
                    |g'(x*)| ≈ {results['derivative_abs']:.6f}  
                    {results['convergence_status']}
                    """)
            
            # Display detailed solution steps
            st.subheader("Solution Steps")
            steps = create_solution_steps(results, "standalone")
            
            for step in steps:
                with st.expander(step["title"]):
                    st.markdown(step["content"])
            
            # Display iteration table
            st.subheader("Iteration Table")
            
            # Create iteration table data
            table_data = []
            for i, x in enumerate(results['x_values']):
                if i == 0:
                    table_data.append([i, f"{x:.10f}", "N/A", f"{results['function_values'][i]:.10e}"])
                else:
                    error = results['iteration_errors'][i-1]
                    table_data.append([i, f"{x:.10f}", f"{error:.10e}", f"{results['function_values'][i]:.10e}"])
            
            df = pd.DataFrame(table_data, columns=["Iteration", "Approximation", "|x_n - x_{n-1}|", "f(x_n)"])
            st.dataframe(df)
            
            # Plot convergence
            st.subheader("Convergence Plots")
            fig_col1, fig_col2 = st.columns(2)
            
            with fig_col1:
                # Approximation convergence plot
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(range(len(results['x_values'])), results['x_values'], 'bo-', label='Approximations')
                ax.axhline(y=results['final_approximation'], color='r', linestyle='--', 
                         label=f"Solution: {results['final_approximation']:.6f}")
                ax.grid(True)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Approximation')
                ax.set_title('Convergence of Fixed Point Iteration')
                ax.legend()
                st.pyplot(fig)
            
            with fig_col2:
                if len(results['iteration_errors']) > 0:
                    # Error convergence plot (log scale)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.semilogy(range(1, len(results['x_values'])), results['iteration_errors'], 'ro-', label='Error')
                    ax.grid(True)
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Error (log scale)')
                    ax.set_title('Error Convergence')
                    ax.legend()
                    st.pyplot(fig)

# Sidebar for API key and file upload
with st.sidebar:
    st.title("Configuration")
    
    # API Key input
    api_key = st.text_input("Enter your Groq API Key:", value=st.session_state.groq_api_key, type="password")
    if api_key:
        st.session_state.groq_api_key = api_key
    
    st.divider()
    st.subheader("Upload Files")
    
    # Text file upload
    uploaded_file = st.file_uploader("Upload a document (PDF, TXT, or code files):", type=["pdf", "txt", "py", "java", "cpp", "html", "css", "js"])
    if uploaded_file is not None:
        content = process_uploaded_file(uploaded_file)
        st.session_state.file_content = content
        st.success(f"Successfully processed {uploaded_file.name}")
        
        if st.button("Initialize Conversation with Document"):
            with st.spinner("Creating vector database from document..."):
                vector_store = create_vectors_from_text(content)
                st.session_state.conversation = initialize_conversation_chain(st.session_state.groq_api_key, vector_store)
            st.success("Conversation initialized with document content!")
    
    st.divider()
    
    # Image upload
    st.subheader("Upload Image")
    uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = handle_image_upload(uploaded_image)
    
    st.divider()
    
    # Audio upload
    st.subheader("Audio Input")
    uploaded_audio = st.file_uploader("Upload an audio file:", type=["wav"])
    if uploaded_audio is not None:
        transcription = handle_audio_upload(uploaded_audio)
        if transcription and st.button("Send Transcription"):
            st.session_state.messages.append({"role": "user", "content": transcription})
    
    # Audio recording option
    if st.button("Record Audio"):
        with st.spinner("Recording audio for 5 seconds..."):
            # Placeholder for audio recording (would require JavaScript integration)
            st.info("Audio recording feature requires JavaScript integration. Use the file upload instead.")
    
    st.divider()
    
    # Initialize conversation without document
    if st.button("Initialize Simple Conversation (No Document)"):
        if st.session_state.groq_api_key:
            st.session_state.conversation = initialize_conversation_chain(st.session_state.groq_api_key)
            st.success("Simple conversation initialized!")
        else:
            st.error("Please enter your Groq API key first!")

# Main page
st.title("Team Avenger's Agent")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Chat Assistant", "Fixed Point Solver"])

with tab1:
    # Display conversation
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If this is an assistant message and there are plots for it, display them
            if message["role"] == "assistant" and i in st.session_state.chat_plots:
                st.subheader("Solution Visualization")
                
                # Display plots side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'convergence' in st.session_state.chat_plots[i]:
                        st.image(st.session_state.chat_plots[i]['convergence'], caption="Convergence Plot")
                
                with col2:
                    if 'error' in st.session_state.chat_plots[i]:
                        st.image(st.session_state.chat_plots[i]['error'], caption="Error Plot")
            
            # If this is an assistant message and there are solution steps, display them
            if message["role"] == "assistant" and i in st.session_state.solution_steps:
                st.subheader("Detailed Solution Steps")
                
                for step in st.session_state.solution_steps[i]:
                    with st.expander(step["title"]):
                        st.markdown(step["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about mathematics, algorithms, or upload content"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if the prompt is asking to solve an equation using fixed point iteration
        is_equation_query, equation_str = detect_equation_query(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_idx = len(st.session_state.messages)
            
            if is_equation_query and equation_str:
                # Extract parameters if specified
                x0, tol, max_iter = extract_parameters_from_prompt(prompt)
                
                with st.spinner("Solving equation using fixed point iteration..."):
                    # First, a brief explanation response
                    intro_response = "I'll solve this equation using the fixed point iteration method."
                    if equation_str != prompt:
                        intro_response += f" I've identified the equation as: `{equation_str}`"
                    
                    intro_response += f"\n\nUsing initial guess x₀ = {x0}, tolerance = {tol:.1e}, and maximum {max_iter} iterations."
                    
                    # Solve the equation
                    results = solve_fixed_point_equation(equation_str, x0, tol, max_iter)
                    
                    if "error" in results:
                        ai_response = f"{intro_response}\n\n❌ {results['error']}"
                    else:
                        # Create plots and store them in session state
                        plots = create_iteration_plots(results, message_idx)
                        
                        # Create solution steps and store them in session state
                        steps = create_solution_steps(results, message_idx)
                        
                        # Build detailed solution response
                        solution_response = f"{intro_response}\n\n"
                        solution_response += f"### Solution Results\n"
                        solution_response += f"**Original equation:** {results['original_equation']}\n"
                        solution_response += f"**Fixed-point form:** {results['fixed_point_form']}\n\n"
                        solution_response += f"**Final approximation:** {results['final_approximation']:.10f}\n"
                        solution_response += f"**Function value at solution:** {results['function_value']:.10e}\n"
                        solution_response += f"**Iterations required:** {results['iterations']}\n"
                        solution_response += f"**Converged:** {results['converged']}\n"
                        
                        if results['converged'] and "derivative_abs" in results:
                            solution_response += f"\n**Convergence analysis:**\n"
                            solution_response += f"|g'(x*)| ≈ {results['derivative_abs']:.6f}\n"
                            solution_response += f"{results['convergence_status']}\n"
                        
                        # Add visualization note
                        solution_response += "\n\nI've created detailed solution steps and visualizations showing how the approximation converged to the solution."
                        
                        ai_response = solution_response
            else:
                # Regular conversation through LLM
                with st.spinner("Thinking..."):
                    if st.session_state.conversation is not None:
                        if isinstance(st.session_state.conversation, ConversationalRetrievalChain):
                            response = st.session_state.conversation({"question": prompt})
                            ai_response = response["answer"]
                        else:
                            # Handle agent or simple LLMChain
                            try:
                                if hasattr(st.session_state.conversation, 'run'):
                                    # For regular LLMChain
                                    response = st.session_state.conversation.run(input=prompt)
                                    ai_response = response
                                else:
                                    # For agent expecting chat_history
                                    chat_history = []
                                    for msg in st.session_state.messages:
                                        if msg["role"] == "user":
                                            chat_history.append(HumanMessage(content=msg["content"]))
                                        elif msg["role"] == "assistant":
                                            chat_history.append(AIMessage(content=msg["content"]))
                                    
                                    # Agent call with chat_history
                                    response = st.session_state.conversation({"input": prompt, "chat_history": chat_history})
                                    ai_response = response["output"] if "output" in response else response
                            except Exception as e:
                                ai_response = f"Error processing your request: {str(e)}"
                    else:
                        ai_response = "Please initialize the conversation first using the sidebar options."
            
            st.markdown(ai_response)
            
            # Text-to-speech option
            if st.button("Listen to Response", key=f"tts_{message_idx}"):
                with st.spinner("Generating audio..."):
                    audio_path = text_to_speech(ai_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

with tab2:
    fixed_point_solver_ui()

# Clean up temporary audio files when the app is closed
def cleanup():
    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        os.unlink(st.session_state.audio_path)

# Register cleanup function to run when app shuts down
import atexit
atexit.register(cleanup)