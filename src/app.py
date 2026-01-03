
import os 

#####################################
##  BitsAndBytes
#####################################

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.llms import HuggingFaceHub
 
def load_model():

    model =  HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"max_length": 1048, "temperature":0.2, "max_new_tokens":512, "top_p":0.95, "repetition_penalty":1.0},
    )

 
    return model

##################################################
## vs chat
##################################################
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
 
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.faiss import FAISS


from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

from langchain_community.document_loaders import TextLoader

def load_txt(path="./a.cv.ckaller.2024.txt"):
    loader = TextLoader(path)
    document = loader.load()
   # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    ####### 
    ''' 
        FAISS
        A FAISS vector store containing the embeddings of the text chunks.
   '''
    model = "BAAI/bge-base-en-v1.5"
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model, encode_kwargs=encode_kwargs, model_kwargs={"device": "cpu"}
    )
    # load from disk
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
 
    #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store = Chroma.from_documents(document_chunks, embeddings, persist_directory="./chroma_db")
 



    #######
    # create a vectorstore from the chunks

    return vector_store


def get_vectorstore_from_url(url="https://huggingface.co/Chris4K"):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    ####### 
    ''' 
        FAISS
        A FAISS vector store containing the embeddings of the text chunks.
   '''
    model = "BAAI/bge-base-en-v1.5"
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model, encode_kwargs=encode_kwargs, model_kwargs={"device": "cpu"}
    )
    # load from disk
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
 
    #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store = Chroma.from_documents(document_chunks, embeddings, persist_directory="./chroma_db")
 



    #######
    # create a vectorstore from the chunks

    return vector_store





def get_context_retriever_chain(vector_store):
 
     

    llm = load_model( )
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = load_model( )
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Du bist Melanie eine Expertin aus Issum. Du weißt alles über Issum. Bist bist lieb, nett und außert höfflich. Du beantwortest gerne alle Fragen bestmöglich basierend auf dem Context. Benutze nur den Inhalt des Context. Füge wenn möglich die Quelle hinzu. Antworte mit: Ich bin mir nicht sicher. Wenn die Antwort nicht aus dem Context hervorgeht. Antworte auf Deutsch. CONTEXT:\n\n{context}"), 
        MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response



###################

###################
import gradio as gr

##from langchain_core.runnables.base import ChatPromptValue
#from torch import tensor

# Create Gradio interface
#vector_store = None  # Set your vector store here
chat_history = []     # Set your chat history here

# Define your function here
def get_response(user_input):

      # Define the prompt as a ChatPromptValue object
    #user_input = ChatPromptValue(user_input)
    
    # Convert the prompt to a tensor
    #input_ids = user_input.tensor
    

    #vs = get_vectorstore_from_url(user_url, all_domain)
    vs = get_vectorstore_from_url()
   # print("------ here 22 " )
    chat_history =[]
    retriever_chain = get_context_retriever_chain(vs)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    #print("get_response " +response)
    res = response['answer']
    parts = res.split(" Assistant: ")
    last_part = parts[-1]
    return last_part


def history_to_dialog_format(chat_history: list[str]):
    dialog = []
    if len(chat_history) > 0:
        for idx, message in enumerate(chat_history[0]):
            role = "user" if idx % 2 == 0 else "assistant"
            dialog.append({
                "role": role,
                "content": message,
            })
    return dialog

def get_response(message, history):
    dialog = history_to_dialog_format(history)
    dialog.append({"role": "user", "content": message})

    print(dialog)
      # Define the prompt as a ChatPromptValue object
    #user_input = ChatPromptValue(user_input)
    
    # Convert the prompt to a tensor
    #input_ids = user_input.tensor
    

    #vs = get_vectorstore_from_url(user_url, all_domain)
    vs = get_vectorstore_from_url("https://huggingface.co/Chris4K")
  
    history =[]
    retriever_chain = get_context_retriever_chain(vs)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": history,
        "input": message  + " Assistant: ",
        "chat_message": message + " Assistant: "
    })
    #print("get_response " +response)
    res = response['answer']
    parts = res.split(" Assistant: ")
    last_part = parts[-1]
    return last_part#[-1]['generation']['content']    

    
#####
#vs = load_txt()
#vs = get_vectorstore_from_url("https://www.xing.com/profile/Christof_Kaller/web_profiles")
#vs = get_vectorstore_from_url("https://www.linkedin.com/in/christof-kaller-6b043733/?originalSubdomain=de")
#vs = get_vectorstore_from_url("https://twitter.com/zX14_7")

 

######

########
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

def get_links_from_page(url, visited_urls, domain_links):
    if url in visited_urls:
        return

    if len(visited_urls) > 3:
        return

    visited_urls.add(url)
    print(url)
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        base_url = urlparse(url).scheme + '://' + urlparse(url).netloc
        links = soup.find_all('a', href=True)

        for link in links:
            href = link.get('href')
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)

            if parsed_url.netloc == urlparse(url).netloc:
                domain_links.add(absolute_url)
                get_links_from_page(absolute_url, visited_urls, domain_links)

    else:
        print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")

def get_all_links_from_domain(domain_url):
    visited_urls = set()
    domain_links = set()
    get_links_from_page(domain_url, visited_urls, domain_links)
    return domain_links



    
# Example usage:
domain_url = 'https://www.issum.de/'
links = get_all_links_from_domain(domain_url)
print("Links from the domain:", links)

#########
##Assuming visited_urls is a list of URLs
for url in links:
    vs = get_vectorstore_from_url(url)
 


def simple(text:str):
  return text +" hhhmmm "

app = gr.ChatInterface(
    fn=get_response,
    #fn=simple,
   # inputs=["text"],
   # outputs="text",
    title="Chat with Websites",
    description="Schreibe hier deine Frage rein...",
    #allow_flagging=False
    retry_btn=None,
    undo_btn=None,
    clear_btn=None,
    #bubble_full_width=False,
    #avatar_images=(None, (os.path.join(os.path.abspath(''), "avatar.png"))),
)

app.launch(debug=True, share=True) 