# All the imports that are necessary for the project to work are imported here
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

#then i loaded the dot env variable to import my dot.env file 
from dotenv import load_dotenv
import os
load_dotenv()

# Get the Google API key from the environment variables.
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')


# Create a chat model using Google's Generative AI model.
model =GoogleGenerativeAI(model="gemini-1.5-flash")
embedding =GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load the PDF file. If it fails to load, it will print the error message and exit the program.
try:
    loader =PyPDFLoader("Stock-Investing-101-eBook.pdf")
except Exception as e:
    print("The error your are incountring is caused to file loading=",e)
    exit()

#to create Embedding while loading module 


# uses text spliter to extract from the embedded  data
text_Splitter = CharacterTextSplitter(chunk_size=500,chunk_overlap=100)

#vector Store for temporary 
index_creator = VectorstoreIndexCreator(
    embedding = embedding,
    text_splitter = text_Splitter
)

index =index_creator.from_loaders([loader])
while True:
    human_message = input("Ask any question about stock :? ")
    response = index.query(human_message,llm=model)
    print(response) 