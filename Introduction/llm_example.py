'''from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()
# llm model 
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = llm.invoke("Give me a tweet about today weather in colombo, sri lanka")

print(result)
'''

# Import ChatGroq class from langchain_groq package
# This is used to connect and talk with Groq AI models
from langchain_groq import ChatGroq

# Import load_dotenv function from dotenv package
# This loads environment variables from .env file
from dotenv import load_dotenv


# Load all variables from .env file into Python environment
# Example: GROQ_API_KEY=your_secret_key
load_dotenv()


# Create the LLM (Large Language Model) object
llm = ChatGroq(

    # Name of the Groq model to use
    # llama3-70b-8192 = powerful Meta Llama 3 model
    model="llama-3.3-70b-versatile",

    # Controls creativity/randomness of answers
    # 0 = strict/focused
    # 1 = more creative
    temperature=0.7
)


# Send prompt/question to the AI model
# invoke() means ask the model and wait for response
result = llm.invoke(
    "Give me a tweet about today weather in Colombo, Sri Lanka"
)


# Print only the text content of AI response
# result contains metadata + content
print(result.content)