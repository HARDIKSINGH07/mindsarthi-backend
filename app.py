import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import json
import warnings

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")  # Use environment variable for security

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["https://mindsaarthi-frontend.onrender.com", "http://localhost:8080"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ------------------------ Load GROQ LLM ------------------------
def load_llm():
    """Load the language model (Groq API)."""
    return ChatGroq(temperature=0.3, model_name="llama3-8b-8192", api_key=groq_api_key)

# ------------------------ Mental Health Chatbot Response Function ------------------------
def generate_response(user_query: str):
    """Generate mental health support response using the LLM."""
    
    # Define the prompt for the chatbot to respond empathetically
    prompt_template = PromptTemplate(
        input_variables=["user_query"],
        template="""You are a mental health support chatbot. You must respond to the user's query with empathy, understanding, and support. If the user expresses concern about their mental well-being, offer resources or encourage them to seek professional help.

        User Query:
        {user_query}

        Your Response: in json format
        """
    )
    
    # Load the language model
    llm = load_llm()
    
    # Create the chain with the prompt and the model
    chain = LLMChain(prompt=prompt_template, llm=llm)
    
    # Get the response
    response = chain.run(user_query=user_query)
    
    return response.strip()

# ------------------------ Flask Route for Chatbot API ------------------------
@app.route('/', methods=['GET'])
def home():
    return "Hello from Backend"

@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint for the chatbot to handle POST requests with user queries."""
    
    # Get the user query from the JSON body
    data = request.get_json()
    user_query = data.get('query', '').strip()

    if not user_query:
        return jsonify({"error": "No query provided."}), 400
    
    # Generate a response for the user's query
    response = generate_response(user_query)

    return jsonify({"response": response})

# ------------------------ Main Section ------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)