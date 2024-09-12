import os
import re
import requests
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlite3 import OperationalError
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import numpy as np
import pickle
import pandas as pd
import spacy
import logging
from transformers import pipeline
from langchain.chains import LLMChain, StuffDocumentsChain
from spacy.lang.de.stop_words import STOP_WORDS as GERMAN_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS as ENGLISH_STOP_WORDS
from difflib import get_close_matches
import time
import openai


import os
import hashlib
import json



# Set the options to display all rows and columns
pd.set_option('display.max_rows', None)  # None means show all rows
pd.set_option('display.max_columns', None)  # None means show all columns
pd.set_option('display.max_colwidth', None)




from groq import Groq
# Load the variables from the JSON file

def set_logging():
    with open('settings_logging.json', 'r') as f:
            variables_logging = json.load(f)

    approach_name =  variables_logging['approach_name'] 
    log_files = "log_files/"   

    logging.basicConfig(filename=f'{log_files}_{approach_name}.log', level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    

    print(f"LOGGING IS SET!! the name is {approach_name}" )






with open('choose_pipeline.json', 'r') as f:
    pipeline = json.load(f)
    pipeline = pipeline['pipeline']




if pipeline == "pipeline_1":
    print("pipeline_1 starts")
    # logging.basicConfig(filename='pipeline_FIRST_KEY.log', level=logging.DEBUG, 
    #                 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    with open('variables1.json', 'r') as f:
        variables = json.load(f)


    # Initialize the client based on configuration
    if variables['use_openai']:
        openai.api_key = variables['OPENAI_API_KEY']
        client = openai
    else:
        os.environ['GROQ_API_KEY'] = variables['GROQ_API_KEY']
        client = Groq(max_retries=20)  # Comment out or remove this if not using Groq
        pass



elif pipeline =="pipeline_2":
    # logging.basicConfig(filename='pipeline_SECOND_KEY.log', level=logging.DEBUG, 
    #                 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("pipeline_2 starts")
    with open('variables2.json', 'r') as f:
        variables = json.load(f)

    # Initialize the client based on configuration
    if variables['use_openai']:
        openai.api_key = variables['OPENAI_API_KEY']
        client = openai
    else:
        os.environ['GROQ_API_KEY'] = variables['GROQ_API_KEY']
        client = Groq(max_retries=20)  # Comment out or remove this if not using Groq
        pass



elif pipeline =="pipeline_3":
    # logging.basicConfig(filename='pipeline_THIRD_KEY.log', level=logging.DEBUG, 
    #                 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("pipeline_3 starts")
    with open('variables3.json', 'r') as f:
        variables = json.load(f)


    # Initialize the client based on configuration
    if variables['use_openai']:
        openai.api_key = variables['OPENAI_API_KEY']
        client = openai
    else:
        os.environ['GROQ_API_KEY'] = variables['GROQ_API_KEY']
        client = Groq(max_retries=20)  # Comment out or remove this if not using Groq
        pass



elif pipeline =="pipeline_4":
    # logging.basicConfig(filename='pipeline_THIRD_KEY.log', level=logging.DEBUG, 
    #                 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("pipeline_4 starts")
    with open('variables4.json', 'r') as f:
        variables = json.load(f)



    # Initialize the client based on configuration
    if variables['use_openai']:
        openai.api_key = variables['OPENAI_API_KEY']
        client = openai
    else:
        os.environ['GROQ_API_KEY'] = variables['GROQ_API_KEY']
        client = Groq(max_retries=20)  # Comment out or remove this if not using Groq
        pass


elif pipeline =="pipeline_5":
    # logging.basicConfig(filename='pipeline_THIRD_KEY.log', level=logging.DEBUG, 
    #                 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("pipeline_5 starts")
    with open('variables5.json', 'r') as f:
        variables = json.load(f)


# Initialize the client based on configuration
    if variables['use_openai']:
        openai.api_key = variables['OPENAI_API_KEY']
        client = openai
    else:
        os.environ['GROQ_API_KEY'] = variables['GROQ_API_KEY']
        client = Groq(max_retries=20)  # Comment out or remove this if not using Groq
        pass



os.environ['GROQ_API_KEY'] = variables['GROQ_API_KEY']
#         client = Groq(max_retries=10)  # Comment out or remove this if not using Groq


# Initialize the GPT4AllEmbeddings
gpt4all_embd = GPT4AllEmbeddings(model_name=variables['model_name'])

# Initialize the Chroma database
# Ensure the persist directory exists
if not os.path.exists(variables['persist_directory']):
    os.makedirs(variables['persist_directory'], exist_ok=True)
    print(f"Created directory: {variables['persist_directory']}")

# Check and modify permissions of the persist directory
if not os.access(variables['persist_directory'], os.W_OK):
    print(f"Changing permissions for {variables['persist_directory']}")
    os.chmod(variables['persist_directory'], 0o775)

db = Chroma(
    embedding_function=gpt4all_embd,
    collection_name=variables['collection_name'],
    persist_directory=variables['persist_directory']
)

# Set the chat model and other variables
chat_model = variables['chat_model']
model = variables['model']

# Define the output path and file
output_path_experiments = variables['output_path_experiments']
output_file = output_path_experiments + f"{variables['content_used']}_classify_replies_800_no_duplicates.csv"

#dependend of model and API used
requests_per_minute = 10 #30
tokens_per_minute = 3500 #5000
seconds_per_request = 60 / requests_per_minute





def get_spacy_model(comment, language):
    """ Load the appropriate spaCy model based on the detected language of the comment. """
    if language == 'DE':
        return spacy.load("de_core_news_sm")
    elif language == 'EN':
        return spacy.load("en_core_web_sm")
    else:
        raise ValueError(f"Unsupported language: {language}")

def load_keywords_from_excel(file_path):
    """
    Load custom keywords from an Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        list: A list of keywords.
    """
    # Load the data from Excel
    df = pd.read_excel(file_path)

    # Assuming keywords are in the column named 'ALL'
    keywords = df['ALL'].tolist()

    # Clean the keywords if necessary (strip whitespace, handle special cases)
    keywords = [str(keyword).strip() for keyword in keywords]
    return keywords




# Metadata file path
metadata_file_path = os.path.join(variables['persist_directory'], 'metadata.json')

# Initialize metadata file if it doesn't exist
if not os.path.exists(metadata_file_path):
    with open(metadata_file_path, 'w') as file:
        json.dump([], file)


# Function to check if an entity already exists in the metadata file
def entity_exists_in_metadata(entity_id):
    with open(metadata_file_path, 'r') as file:
        metadata = json.load(file)
    return entity_id in metadata

# Function to update the metadata file with a new entity
def update_metadata_file(entity_id):
    with open(metadata_file_path, 'r') as file:
        metadata = json.load(file)
    metadata.append(entity_id)
    with open(metadata_file_path, 'w') as file:
        json.dump(metadata, file)


# Function to process and store entities
def process_and_store(text, entity_id):
    if entity_exists_in_metadata(entity_id):
        print(f"Entity {entity_id} already exists in the metadata. Skipping.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        gpt_vector = gpt4all_embd.embed_query(chunk)
        try:
            # Add document and embedding directly to the database
            db.add_texts(
                [chunk],
                ids=[f"{entity_id}_part_{i}"],
                embeddings=[gpt_vector]
            )
            print(f"Content added to database: {entity_id}_part_{i}")
        except Exception as e:
            print(f"Error: {e}. Please check the database file permissions.")
    
    # Update the metadata file after successfully adding the entry to the database
    update_metadata_file(entity_id)

# Example usage
process_and_store("Sample text content", "entity_123")




def score_entities(entities, comment):
    """Score entities based on attention scoring using a multilingual transformer model.
    --> compute a relative similarity score for each entity in a list of entities compared to a given comment (with cosine similarity)."""
    comment_embedding = model.encode([comment])[0]
    entity_scores = {}

    for entity in entities:
        entity_embedding = model.encode([entity])[0]
        similarity = np.dot(comment_embedding, entity_embedding) / (
            np.linalg.norm(comment_embedding) * np.linalg.norm(entity_embedding)
        )
        entity_scores[entity] = similarity

    max_score = max(entity_scores.values()) if entity_scores else 1
    for entity in entity_scores:
        entity_scores[entity] /= max_score
    return entity_scores

def compute_tfidf(corpus):
    """Compute TF-IDF scores for a given corpus."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()
    return feature_names, tfidf_scores

def get_most_important_entity(entities, text, corpus):
    """Get the most contextually important entity from the text based on TF-IDF scores."""
    # Create a corpus with the current text and additional context
    extended_corpus = corpus + [text]

    # Compute TF-IDF scores for the extended corpus
    feature_names, tfidf_scores = compute_tfidf(extended_corpus)

    # Extract the scores for the last document (the current text)
    entity_scores = {}
    for entity in entities:
        if entity in feature_names:
            idx = feature_names.tolist().index(entity)
            entity_scores[entity] = tfidf_scores[-1][idx]

    # Find the entity with the highest TF-IDF score
    if entity_scores:
        most_important_entity = max(entity_scores, key=entity_scores.get)
        return entity_scores
    else:
        return {ent: 1 for ent in entities}
    


###################################### ENGLSIH classify_reply ######################################################

def EN_classify_reply(reply, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}. 
            Comment: '{reply}' 
            Question: What is the classification of the comment? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply']
    )

    #print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content


###################################### ENGLSIH classify_reply_title ######################################################

def EN_classify_reply_title(reply, title, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}. 
            Video title: '{title}'
            Comment: '{reply}' 
            Question: What is the classification of the comment? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'title']
    )

    #print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        title=title
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content

###################################### ENGLSIH classify_reply_comment ######################################################

def EN_classify_reply_comment(reply, comment, classes):

    print("COMMENT: ", comment)
    print("REPLY: ", reply)

    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}. 
            Precending comment: '{comment}' 
            Reply comment: '{reply}' 
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment']
    )

    #print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content



###################################### ENGLSIH classify_reply_comment_title ######################################################

def EN_classify_reply_comment_title(reply, comment, title, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}' 
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'title']
    )

    #print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content


###################################### ENGLSIH classify_reply_comment_title_definition ######################################################

def EN_classify_reply_comment_title_definition(reply, comment, title, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}' 
            Here are the definitions of the classes to choose from: 
                1. Appropriate - no target 
                2. Inappropriate (contains terms that are obscene, vulgar; but the text is not directed at
                any person specifically) - has no target 
                3. Offensive (including offensive generalization, contempt, dehumanization, indirect
                offensive remarks)
                4. Violent (author threatens, indulges, desires or calls for physical violence against a
                target; it also includes calling for, denying or glorifying war crimes and crimes against
                humanity)

            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'title']
    )

    #print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content



###################################### ENGLSIH classify_reply_comment_title_definition_transcript ######################################################

def EN_classify_reply_comment_title_definition_transcript(reply, comment, title, transcript, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}' 
            Transcript of the video: '{transcript}' 
            Here are the definitions of the classes to choose from: 
                1. Appropriate - no target 
                2. Inappropriate (contains terms that are obscene, vulgar; but the text is not directed at
                any person specifically) - has no target 
                3. Offensive (including offensive generalization, contempt, dehumanization, indirect
                offensive remarks)
                4. Violent (author threatens, indulges, desires or calls for physical violence against a
                target; it also includes calling for, denying or glorifying war crimes and crimes against
                humanity)

            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'title', 'transcript']
    )

    #print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title,
        transcript = transcript,
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content





###################################### ENGLSIH classify_reply_comment_title_definition_transcript ######################################################
def EN_classify_reply_comment_title_definition_description_descriptionYT_transcript_audio_event_object(reply, comment, video_title, video_description, descriptionYT, transcript, audio_event, object, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """
    

    PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{video_title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Description of the video: '{video_description}'
            Description posted under the video: '{descriptionYT}'
            Transcript of the video: '{transcript}'
            Audio events and occurence scores of the video: '{audio_event}'
            Objects shown in the video: '{object}'
            Here are the definitions of the classes to choose from: 
                1. Appropriate - no target 
                2. Inappropriate (contains terms that are obscene, vulgar; but the text is not directed at
                any person specifically) - has no target 
                3. Offensive (including offensive generalization, contempt, dehumanization, indirect
                offensive remarks)
                4. Violent (author threatens, indulges, desires or calls for physical violence against a
                target; it also includes calling for, denying or glorifying war crimes and crimes against
                humanity)

            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply','comment', 'video_title', 'video_description', 'descriptionYT', 'transcript', 'audio_event', 'object']
    )

    # Build the full prompt using all provided context
    prompt = PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH.format(
        comment=comment,
        reply=reply,
        video_title=video_title,
        video_description=video_description,
        descriptionYT=descriptionYT,
        transcript=transcript,
        audio_event=audio_event,
        object=object
    )


    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content


###################################### ENGLSIH classify_reply_title_description ######################################################

def EN_classify_reply_title_description(reply, title, description, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Description of the video: '{description}' 
            Comment: '{reply}' 
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'title', 'description']
    )

    #print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        title = title,
        description = description
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content



###################################### ENGLSIH classify_reply_comment_title_description ######################################################

def EN_classify_reply_comment_title_description_OLD(reply, comment, title, description, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Description of the video: '{description}'
            Precending comment: {comment}' 
            Reply comment: '{reply}' 
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'title', 'description']
    )

    #print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title,
        description = description
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content




###################################### ENGLSIH classify_reply_comment_title_description_prompt2 ######################################################

def EN_classify_reply_comment_title_description_prompt2(reply, comment, title, description, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Precending comment: {comment}' 
            Reply comment: '{reply}' 
            Here is some additional information so you can understand the context of the comments better:
            Video title: '{title}'
            Description of the video: '{description}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'title', 'description']
    )

    #print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title,
        description = description
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content





###################################### ENGLSIH classify_reply_comment_title_transcript ######################################################

def EN_classify_reply_comment_title_transcript(reply, comment, title, transcript, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Transcript of the video: '{transcript}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'title', 'transcript']
    )

    print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title,
        transcript = transcript
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content

###################################### ENGLSIH classify_reply_comment_title_transcript_audio_event ######################################################

def EN_classify_reply_comment_title_transcript_audio_event(reply, comment, title, transcript, audio_event, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Transcript of the video: '{transcript}'
            Audio events and occurence scores of the video: '{audio_event}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'title', 'transcript', 'audio_event']
    )

    print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title,
        transcript = transcript,
        audio_event = audio_event
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content



###################################### ENGLSIH classify_reply_comment_title_transcript_object ######################################################

def EN_classify_reply_comment_title_transcript_object(reply, comment, title, transcript, object, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Transcript of the video: '{transcript}'
            Objects shown in the video: '{object}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'title', 'transcript', 'object']
    )

    print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title,
        transcript = transcript,
        object = object
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content


###################################### ENGLSIH classify_reply_comment_title_metadata_video_reply ######################################################

def EN_classify_reply_comment_title_metadata_video_reply(reply, comment, title, view_count_video, like_count_video, favorite_count_video, comment_count_video, tags_video, channel_title_video, author_reply, like_count_reply, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Number of views on the video: '{view_count_video}'
            Number of likes on the video: '{like_count_video}'
            Number of favorites on the video: '{favorite_count_video}'
            Number of comments of the video: '{comment_count_video}'
            Used tags on the video: '{tags_video}'
            Channel Title that posted the video : '{channel_title_video}
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Name of the author of the reply comment: '{author_reply}'
            Number of likes on reply comment: '{like_count_reply}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'title', 'view_count_video', 'like_count_video', 'comment_count_video', 'tags_video', 'channel_title_video', 'author_reply', 'like_count_reply']
    )

    print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title,
        view_count_video = view_count_video,
        like_count_video = like_count_video,
        favorite_count_video = favorite_count_video,
        comment_count_video = comment_count_video,
        tags_video = tags_video,
        channel_title_video = channel_title_video,
        author_reply = author_reply,
        like_count_reply= like_count_reply
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content




###################################### ENGLSIH classify_reply_comment_title_transcript_description ######################################################

def EN_classify_reply_comment_title_transcript_description(reply, comment, title, transcript, description, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Transcript of the video: '{transcript}'
            Description of the video: '{description}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'title', 'transcript', 'description']
    )

    print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title,
        transcript = transcript,
        description = description
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content



def EN_classify_reply_comment_transcript(reply, comment, transcript, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Transcript of the video: '{transcript}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'comment', 'transcript']
    )

    print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        transcript = transcript
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content


###################################### ENGLSIH classify_reply_title_transcript ######################################################

def EN_classify_reply_title_transcript(reply, title, transcript, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Comment: '{reply}'
            Transcript of the video: '{transcript}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'title', 'transcript']
    )

    print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        title = title,
        transcript = transcript
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content


###################################### ENGLSIH classify_reply_title_transcript_description ######################################################

def EN_classify_reply_title_transcript_description(reply, title, transcript, description, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Comment: '{reply}'
            Transcript of the video: '{transcript}'
            Description of the video: '{description}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'title', 'transcript', 'description']
    )

    print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        title = title,
        transcript = transcript,
        description = description
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content

###################################### ENGLSIH classify_reply_comment_title_descriptionYT ######################################################

def EN_classify_reply_comment_title_descriptionYT(reply, comment, title, descriptionYT, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Description posted under the video: '{descriptionYT}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'title', 'descriptionYT']
    )

    print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        comment = comment,
        title = title,
        descriptionYT = descriptionYT
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content


###################################### ENGLSIH classify_reply_transcript ######################################################

def EN_classify_reply_transcript(reply, transcript, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_COMMENT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Comment: '{reply}'
            Transcript of the video: '{transcript}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'title', 'transcript']
    )

    print("PROMPT TEMPLATE: ", PROMPT_COMMENT_TEMPLATE_ENGLISH)

    # Build the full prompt using all provided context
    prompt = PROMPT_COMMENT_TEMPLATE_ENGLISH.format(
        reply=reply,
        transcript = transcript
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content




###################################### ENGLSIH classify_reply_comment_title_external_context ######################################################

def EN_classify_reply_comment_title_external_context(reply, comment, title, context_documents, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """
    enriched_context = '\n'.join(context_documents) #join list to one string


    PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            External context: '{context_documents}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply','comment', 'title', 'external_context']
    )

    # Build the full prompt using all provided context
    prompt = PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH.format(
        comment=comment,
        reply=reply,
        title=title,
        external_context=enriched_context
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content


###################################### ENGLSIH classify_reply_title_external_context ######################################################

def EN_classify_reply_title_external_context(reply,  title, context_documents, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """
    enriched_context = '\n'.join(context_documents) #join list to one string


    PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{title}'
            Reply comment: '{reply}'
            External context: '{context_documents}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply', 'title', 'external_context']
    )

    # Build the full prompt using all provided context
    prompt = PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH.format(
        reply=reply,
        title=title,
        external_context=enriched_context
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content



###################################### ENGLSIH classify_reply_comment_title_transcript_external_context ######################################################

def EN_classify_reply_comment_title_transcript_external_context(reply, comment, video_title, transcript, context_documents, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """
    enriched_context = '\n'.join(context_documents) #join list to one string


    PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{video_title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Transcript of the video: '{transcript}'
            External context: '{context_documents}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply','comment', 'video_title', 'transcript', 'external_context']
    )

    # Build the full prompt using all provided context
    prompt = PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH.format(
        comment=comment,
        reply=reply,
        video_title=video_title,
        transcript=transcript,
        external_context=enriched_context
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content



###################################### ENGLSIH classify_classify_reply_comment_title_description_descriptionYT_external_context_transcript_audio_event_object ######################################################

def EN_classify_reply_comment_title_description_descriptionYT_external_context_transcript_audio_event_object(reply, comment, video_title, video_description, descriptionYT, transcript, audio_event, object, context_documents, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """
    enriched_context = '\n'.join(context_documents) #join list to one string


    PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{video_title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Description of the video: '{video_description}'
            Description posted under the video: '{descriptionYT}'
            Transcript of the video: '{transcript}'
            Audio events and occurence scores of the video: '{audio_event}'
            Objects shown in the video: '{object}'
            External context: '{context_documents}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply','comment', 'video_title', 'video_description', 'descriptionYT', 'transcript', 'audio_event', 'object', 'external_context']
    )

    # Build the full prompt using all provided context
    prompt = PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH.format(
        comment=comment,
        reply=reply,
        video_title=video_title,
        video_description=video_description,
        descriptionYT=descriptionYT,
        transcript=transcript,
        audio_event=audio_event,
        object=object,
        external_context=enriched_context
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content



###################################### ENGLSIH classify_reply_comment_title_description_descriptionYT_transcript_audio_event_object ######################################################

def EN_classify_reply_comment_title_description_descriptionYT_transcript_audio_event_object(reply, comment, video_title, video_description, descriptionYT, transcript, audio_event, object, classes):
    """
    Classify a comment using context from the video, transcript, retrieved documents, and a reply.
    """

    PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH = PromptTemplate(
        template=(
            f"""You are a model designed to classify video comments as {', '.join(classes)}.  
            Video title: '{video_title}'
            Precending comment: {comment}' 
            Reply comment: '{reply}'
            Description of the video: '{video_description}'
            Description posted under the video: '{descriptionYT}'
            Transcript of the video: '{transcript}'
            Audio events and occurence scores of the video: '{audio_event}'
            Objects shown in the video: '{object}'
            Question: What is the classification of the reply? 
            Please classify the comment and provide an explanation in the following format: 'Classification': '', 'Explanation': ''"""
        ),
        input_variables=['reply','comment', 'video_title', 'video_description', 'descriptionYT', 'transcript', 'audio_event', 'object']
    )

    # Build the full prompt using all provided context
    prompt = PROMPT_WITH_EXTERNAL_CONTEXT_TEMPLATE_ENGLISH.format(
        comment=comment,
        reply=reply,
        video_title=video_title,
        video_description=video_description,
        descriptionYT=descriptionYT,
        transcript=transcript,
        audio_event=audio_event,
        object=object
    )

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": f"You are a model designed to classify comments as {', '.join(classes)}."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=chat_model,  # Ensure this is a string
        temperature=1,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content





# # Function to correct the format of the response
# def correct_response_format(response):
#     if not response:
#         return None, None
    
#     # Check if the response is in dictionary format
#     if isinstance(response, dict):
#         classification = response.get('Classification')
#         explanation = response.get('Explanation')
#         return classification, explanation
    
#     # Regular expressions to match the classification and explanation
#     class_pattern = r"'?Classification'?: '?([a-zA-Z ]+)'?"
#     explanation_pattern = r"'?Explanation'?: '?(.+)'?"
    
#     # Search for the patterns in the response
#     class_match = re.search(class_pattern, response)
#     explanation_match = re.search(explanation_pattern, response)
    
#     # Extract the classification and explanation if matches are found
#     if class_match and explanation_match:
#         classification = class_match.group(1)
#         explanation = explanation_match.group(1)
#         return classification, explanation
    
#     # If the response doesn't match the expected format, return None
#     return None, None


def process_comment(video_title, transcript, video_description, descriptionYT, audio_event, object, view_count_video, like_count_video, favorite_count_video, comment_count_video, tags_video, channel_title_video, author_reply, like_count_reply, comment, reply, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords, content_used, classes):
    # #logging.info('Processing comment')
    # logging.debug(f'Reply: {reply}')
    # logging.debug(f'Comment: {comment}')

    print("you are in the process comment function!")
    print(f'Reply: {reply}')
    print(f'Comment: {comment}')


    


   

    #______________GERMAN (DE)______________________________________________________
    if content_used == "DE_comment":
        print('content_used == "DE_comment"')
        response = DE_classify_comment_title_comment(comment, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        print(f"Direct LLM Response (No Context): {response}")


    if content_used == "DE_title_comment":
        print('content_used == "DE_title_comment"')
        response = DE_classify_comment_title_comment(video_title, comment, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        print(f"Direct LLM Response (No Context): {response}")
    

    elif content_used == "DE_title_transcript_comment":
        print('content_used == "DE_title_transript_comment"')
        response = DE_classify_comment_title_transcript_comment(video_title, transcript, comment, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        print(f"Direct LLM Response (No Context): {response}")
    

    elif content_used == "DE_title_transcript_comment_reply":
        print('content_used == "DE_title_transript_comment_reply"')
        response = DE_classify_comment_title_transcript_comment_reply(video_title, transcript, comment, reply, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        print(f"Direct LLM Response (No Context): {response}")


    elif content_used == "DE_title_transcript_comment_reply_external_context":
        print('content_used == "DE_title_transript_comment_reply_external_context"')
        context = process_entities_from_comment(video_title, comment, reply, transcript, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords)
        #logging.debug(f'context short: {context}')
        #print(f"context short: {context}")
    
        context_database = retrieve_context_for_comment(comment)
        print("USED CONTEXT FROM DATABASE: ", context_database)
        response = DE_classify_comment_title_transcript_comment_reply_external_context(video_title, transcript, comment, reply, context_database, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        print(f"LLM Response with Context: {response}")


    ###################
    elif content_used == "DE_title_transcript_video_description_comment_reply_external_context":
        print('content_used == "DE_title_transcript_video_description_comment_reply_external_context"')
        response = DE_classify_title_transcript_video_description_comment_reply_external_context(video_title, transcript, video_description, comment, reply, context_database, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        print(f"Direct LLM Response (No Context): {response}")

    elif content_used == "DE_title_video_description_comment_reply":
        print('content_used == "DE_title_video_description_comment_reply"')
        response = DE_classify_title_video_description_comment_reply(video_title, video_description, comment, reply, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        print(f"Direct LLM Response (No Context): {response}")

    elif content_used == "DE_title_transcript_video_description_comment_reply_external_context":
        print('content_used == "DE_title_transcript_video_description_comment_reply_external_context"')
        context = process_entities_from_comment(video_title, comment, reply, transcript, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords)
        logging.debug(f'context short: {context}')
        #print(f"context short: {context}")
    
        context_database = retrieve_context_for_comment(comment)
        print("USED CONTEXT FROM DATABASE: ", context_database)
        response = DE_classify_title_transcript_video_description_comment_reply_external_context(video_title, transcript, video_description, audio_events, comment, reply, context_database, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        print(f"LLM Response with Context: {response}")




    #_____________ENGLISH (EN)__________________________________________________________

    if content_used == "EN_reply":
        print('content_used == "EN_reply"')
        response = EN_classify_reply(reply, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")

    if content_used == "EN_reply_title":
        print('content_used == "EN_reply_title"')
        response = EN_classify_reply_title(reply, video_title, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")
    
    if content_used == "EN_reply_comment":
        print('content_used == "EN_reply_comment"')
        response =  EN_classify_reply_comment(reply, comment, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")

    if content_used == "EN_reply_comment_title":
        print('content_used == "EN_reply_comment_title"')
        response =  EN_classify_reply_comment_title(reply, comment, video_title, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")

    if content_used == "EN_reply_comment_title_definition":
        print('content_used == "EN_reply_comment_title_definition"')
        response =  EN_classify_reply_comment_title_definition(reply, comment, video_title, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")


    if content_used == "EN_reply_comment_title_definition_transcript":
        print('content_used == "EN_reply_comment_title_definition_transcript"')
        response =  EN_classify_reply_comment_title_definition_transcript(reply, comment, video_title, transcript, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")

    if content_used == "EN_classify_reply_comment_title_definition_description_descriptionYT_transcript_audio_event_object":
        print('content_used == "EN_classify_reply_comment_title_definition_description_descriptionYT_transcript_audio_event_object"')
        response =  EN_classify_reply_comment_title_definition_description_descriptionYT_transcript_audio_event_object(reply, comment, video_title, video_description, descriptionYT, transcript, audio_event, object, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")

    



    if content_used == "EN_reply_comment_title_description":
        print('content_used == "EN_reply_comment_title_description"')
        response =  EN_classify_reply_comment_title_description_OLD(reply, comment, video_title, video_description, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")

    if content_used == "EN_reply_title_description":
        print('content_used == "EN_reply_title_description"')
        response =  EN_classify_reply_title_description(reply,  video_title, video_description, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")
    


    if content_used == "EN_reply_comment_title_description_prompt2":
        print('content_used == "EN_reply_comment_title_description_prompt2"')
        response =  EN_classify_reply_comment_title_description(reply, comment, video_title, video_description, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")
    

    if content_used == "EN_reply_comment_title_transcript":
        print('content_used == "EN_reply_comment_title_transcript"')
        response =  EN_classify_reply_comment_title_transcript(reply, comment, video_title, transcript, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")

    if content_used == "EN_reply_comment_title_transcript_audio_event":
        print('content_used == "EN_reply_comment_title_transcript_audio_event"')
        response =  EN_classify_reply_comment_title_transcript_audio_event(reply, comment, video_title, transcript, audio_event, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")


    if content_used == "EN_reply_comment_title_transcript_object":
        print('content_used == "EN_reply_comment_title_transcript_object"')
        response =  EN_classify_reply_comment_title_transcript_object(reply, comment, video_title, transcript, object, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")



    if content_used == "EN_reply_comment_title_transcript_description":
        print('content_used == "EN_reply_comment_title_transcript_description"')
        response =  EN_classify_reply_comment_title_transcript_description(reply, comment, video_title, transcript, video_description, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")



    if content_used == "EN_reply_comment_transcript":
        print('content_used == "EN_reply_comment_transcript"')
        response =  EN_classify_reply_comment_transcript(reply, comment,  transcript, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")

    if content_used == "EN_reply_title_transcript":
        print('content_used == "EN_reply_title_transcript"')
        response =  EN_classify_reply_title_transcript(reply,  video_title, transcript, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")


    if content_used == "EN_reply_title_transcript_description":
        print('content_used == "EN_reply_title_transcript_description"')
        response =  EN_classify_reply_title_transcript_description(reply,  video_title, transcript, video_description, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")

    if content_used == "EN_reply_comment_title_descriptionYT":
        print('content_used == "EN_reply_comment_title_descriptionYT"')
        response =  EN_classify_reply_comment_title_descriptionYT(reply, comment, video_title, descriptionYT, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"Direct LLM Response (No Context): {response}")


    if content_used == "EN_reply_transcript":
        print('content_used == "EN_reply_transcript"')
        response =  EN_classify_reply_transcript(reply,  transcript, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")

    if content_used == "EN_reply_comment_title_metadata_video_reply":
        print('content_used == "EN_reply_comment_title_metadata_video_reply')
        response =  EN_classify_reply_comment_title_metadata_video_reply(reply, comment, video_title, view_count_video, like_count_video, favorite_count_video, comment_count_video, tags_video, channel_title_video, author_reply, like_count_reply, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")


        
    if content_used == "EN_classify_reply_comment_title_description_descriptionYT_transcript_audio_event_object":
        print('content_used == "EN_classify_reply_comment_title_description_descriptionYT_transcript_audio_event_object')
        response = EN_classify_reply_comment_title_description_descriptionYT_transcript_audio_event_object(reply, comment, video_title, video_description, descriptionYT, transcript, audio_event, object, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")



    if content_used == "EN_reply_comment_title_external_context":
            print('content_used == "EN_reply_comment_title_description_external_context"')
            context = process_entities_from_comment(video_title, comment, reply, transcript, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords)
            logging.debug(f'context short: {context}')
            #print(f"context short: {context}")
            info = f"{video_title} {comment} {reply} {transcript}"
            context_database = retrieve_context_for_comment(info)
            #print("USED CONTEXT FROM DATABASE: ", context_database)
            
            response = EN_classify_reply_comment_title_external_context(reply, comment, video_title, context_database, classes)
            
            print("HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERE")
            
            logging.info(f"Direct LLM Response (No Context): {response}")
            #print(f"LLM Response with Context: {response}")

    if content_used == "EN_reply_comment_title_description_descriptionYT_external_context_transcript_audio_event_object":
            print('content_used == "EN_reply_comment_title_description_descriptionYT_external_context_transcript_audio_event_object"')
            print("hallo")
            context = process_entities_from_comment(video_title, comment, reply, transcript, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords)
            logging.debug(f'context short: {context}')
            #print(f"context short: {context}")
            info = f"{video_title} {comment} {reply} {transcript}"
            context_database = retrieve_context_for_comment(info)
            #print("USED CONTEXT FROM DATABASE: ", context_database)
            
            response = EN_classify_reply_comment_title_description_descriptionYT_external_context_transcript_audio_event_object(reply, comment, video_title, video_description, descriptionYT, transcript, audio_event, object, context_database, classes)
            logging.info(f"Direct LLM Response (No Context): {response}")
            #print(f"LLM Response with Context: {response}")



    if content_used == "EN_reply_title_external_context":
            print('content_used == "EN_reply_title_external_context"')
            context = process_entities_from_comment(video_title, comment, reply, transcript, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords)
            logging.debug(f'context short: {context}')
            #print(f"context short: {context}")
            info = f"{video_title} {comment} {reply} {transcript}"
            context_database = retrieve_context_for_comment(info)
            #print("USED CONTEXT FROM DATABASE: ", context_database)
            
            response = EN_classify_reply_title_external_context(reply, video_title, context_database, classes)
            logging.info(f"Direct LLM Response (No Context): {response}")
            #print(f"LLM Response with Context: {response}")




    if content_used == "EN_reply_comment_title_description_external_context":
            print('content_used == "EN_reply_comment_title_description_external_context"')
            context = process_entities_from_comment(video_title, comment, reply, transcript, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords)
            logging.debug(f'context short: {context}')
            #print(f"context short: {context}")
            info = f"{video_title} {comment} {reply} {transcript}"
            context_database = retrieve_context_for_comment(info)
            #print("USED CONTEXT FROM DATABASE: ", context_database)
            
            response = EN_classify_reply_comment_title_description_external_context(reply, comment, video_title, video_description, context_database, classes)
            logging.info(f"Direct LLM Response (No Context): {response}")
            #print(f"LLM Response with Context: {response}")
    
    
    #print("content_used: ", content_used)
    #print("you are here")
    if content_used == "EN_reply_comment_title_transcript_external_context":
        print('content_used == "EN_reply_comment_title_transcript_external_context"')
        context = process_entities_from_comment(video_title, comment, reply, transcript, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords)
        logging.debug(f'context short: {context}')
        #print(f"context short: {context}")
        info = f"{video_title} {comment} {reply} {transcript}"
        context_database = retrieve_context_for_comment(info)
        #print("USED CONTEXT FROM DATABASE: ", context_database)
        
        response = EN_classify_title_transcript_comment_reply_external_context(reply, comment, video_title, transcript, context_database, classes)
        logging.info(f"Direct LLM Response (No Context): {response}")
        #print(f"LLM Response with Context: {response}")
    
# ###########################################################################################    
#     if content_used == "EN_comment":
#         print('content_used == "EN_comment"')
#         response = EN_classify_comment(comment, classes)
#         logging.info(f"Direct LLM Response (No Context): {response}")
#         print(f"Direct LLM Response (No Context): {response}")
    
    
    
#     if content_used == "EN_title_comment":
#         print('content_used == "EN_title_comment"')
#         response = EN_classify_title_comment(video_title, comment, classes)
#         logging.info(f"Direct LLM Response (No Context): {response}")
#         print(f"Direct LLM Response (No Context): {response}")
    

#     elif content_used == "EN_title_transcript_comment":
#         print('content_used == "EN_title_transript_comment"')
#         response = EN_classify_title_transcript_comment(video_title, transcript, comment, classes)
#         logging.info(f"Direct LLM Response (No Context): {response}")
#         print(f"Direct LLM Response (No Context): {response}")
    

#     elif content_used == "EN_title_transcript_comment_reply":
#         print('content_used == "EN_title_transript_comment_reply"')
#         response = EN_classify_title_transcript_comment_reply(video_title, transcript, comment, reply, classes)
#         logging.info(f"Direct LLM Response (No Context): {response}")
#         print(f"Direct LLM Response (No Context): {response}")


#     elif content_used == "EN_title_transcript_comment_reply_external_context":
#         print('content_used == "EN_title_transript_comment_reply_external_context"')
#         context = process_entities_from_comment(video_title, comment, reply, transcript, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords)
#         logging.debug(f'context short: {context}')
#         #print(f"context short: {context}")
    
#         context_database = retrieve_context_for_comment(reply)
#         print("USED CONTEXT FROM DATABASE: ", context_database)
#         response = EN_classify_reply_comment_title_transcript_external_context(video_title, transcript, comment, reply, context_database, classes)
#         logging.info(f"LLM Response with Context: {response}")
#         print(f"LLM Response with Context: {response}")

    # ###################
    # elif content_used == "EN_title_transcript_video_description_comment_reply_external_context":
    #     print('content_used == "EN_title_transcript_video_description_comment_reply_external_context"')
    #     response = EN_classify_title_transcript_video_description_comment_reply_external_context(video_title, transcript, video_description, comment, reply, context_database, classes)
    #     logging.info(f"Direct LLM Response (No Context): {response}")
    #     print(f"Direct LLM Response (No Context): {response}")

    # elif content_used == "EN_title_video_description_comment_reply":
    #     print('content_used == "EN_title_video_description_comment_reply"')
    #     response = EN_classify_title_video_description_comment_reply(video_title, video_description, comment, reply, classes)
    #     logging.info(f"Direct LLM Response (No Context): {response}")
    #     print(f"Direct LLM Response (No Context): {response}")

    # elif content_used == "EN_title_transcript_video_description_comment_reply_external_context":
    #     print('content_used == "EN_title_transcript_video_description_comment_reply_external_context"')
    #     context = process_entities_from_comment(video_title, comment, reply, transcript, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords)
    #     logging.debug(f'context short: {context}')
    #     #print(f"context short: {context}")
    
    #     context_database = retrieve_context_for_comment(comment)
    #     print("USED CONTEXT FROM DATABASE: ", context_database)
    #     response = EN_classify_title_transcript_video_description_comment_reply_external_context(video_title, transcript, video_description, audio_events, comment, reply, context_database, classes)
    #     logging.info(f"LLM Response with Context: {response}")
    #     print(f"LLM Response with Context: {response}")


    # Parse and correct the response
    classification, explanation = correct_response_format(response)#, classes)

    return classification, explanation, comment, content_used, reply #classification, explanation, comment, content_used





def preprocess_response(response):
    response = response.replace("{", "").replace("}", "")
    response = re.sub(r'(["`])', "'", response)  # Replace double quotes and backticks with single quotes
    return response

def classify_response_with_llm(response, model="mixtral-8x7b-32768", temperature=0.8):

    groq_client = Groq(max_retries=20)

    #os.environ['GROQ_API_KEY'] = variables['GROQ_API_KEY']
            



    # Preprocess the response to remove curly braces
    response = response.replace("{", "").replace("}", "")

    PROMPT_TEMPLATE = (
        f"""Extract the classification and explanation from this response: '{response}'
        and format them as:
        'Classification': 'classification',
        'Explanation': 'explanation'
        
        Requirements:
        - Use lowercase for classification.
        - Preserve the original content of the explanation.
        - Use single quotes (') only.
        - No extra characters or formatting.
        - No double quotes (").
        
        Example:
        'Classification': 'example', 'Explanation': 'Here comes the explanation.'
        """
    )

    # Build the prompt
    prompt = PROMPT_TEMPLATE.format(response=response)

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": "You are a model designed to format text correctly."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content

def validate_response_format(response):
    # Pattern to match 'Classification': 'value', 'Explanation': 'value'
    pattern = re.compile(r"'[Cc]lassification':\s*'([^']+)',\s*'[Ee]xplanation':\s*'([^']*)'")
    return bool(pattern.search(response))

def prompt_llm_to_correct_format(response):
    # Preprocess the response to remove curly braces

    groq_client = Groq(max_retries=20)
    response = response.replace("{", "").replace("}", "")

    PROMPT_TEMPLATE = (
        f"""The response format is incorrect. Please correct the format of the following response:
        '{response}'
        
        Ensure the format is:
        'Classification': 'classification',
        'Explanation': 'explanation'
        
        Requirements:
        - Use lowercase for classification.
        - Preserve the original content of the explanation.
        - Use single quotes (') only.
        - No extra characters or formatting.
        - No double quotes (").
        
        Example:
        'Classification': 'example', 'Explanation': 'Here comes the explanation.'
        """
    )

    # Build the prompt
    prompt = PROMPT_TEMPLATE.format(response=response)

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": "You are a model designed to format text correctly."},
        {"role": "user", "content": prompt}
    ]

    # Make the API call with the correct model identifier
    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model='mixtral-8x7b-32768',
        temperature=0.6,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content

def normalize_classification(classification, valid_classes):
    if not classification:  # Ensure classification is not None or empty
        return None
    classification = classification.strip().lower()
    closest_matches = get_close_matches(classification, valid_classes, n=1, cutoff=0.6)
    if closest_matches:
        return closest_matches[0]
    else:
        return classification  # Keep the original classification if no close match is found

def correct_response_format(response):
    if not response:
        return None, None

    logging.debug(f"Original response: {response}")
    print(f"Original response: {response}")

    response = preprocess_response(response)  # Preprocess the response
    count = 1
    max_retries = 1  # Set a limit to avoid infinite loops

    while count <= max_retries:
        # Use the LLM directly
        print("Iteration: ", count)
        llm_response = classify_response_with_llm(response)
        logging.debug(f"LLM response: {llm_response}")
        print(f"LLM response: {llm_response}")

        llm_response = preprocess_response(llm_response)  # Preprocess the LLM response

        # Validate the LLM response format
        if validate_response_format(llm_response):
            # Pattern to match 'Classification': 'value', 'Explanation': 'value'
            pattern = re.compile(r"'Classification':\s*'([^']+)',\s*'Explanation':\s*'([^']*)'")
            
            match = pattern.search(llm_response)
            if match:
                classification = match.group(1).lower()
                explanation = match.group(2).replace("\\'", "'")  # Handle escaped quotes

                # Ensure the explanation is properly quoted
                if not explanation.startswith("'") or not explanation.endswith("'"):
                    explanation = f"'{explanation.strip()}'"

                print(f"Parsed from LLM - classification: {classification}, explanation: {explanation}")
                logging.debug(f"Parsed from LLM - classification: {classification}, explanation: {explanation}")
                return classification, explanation

            raise ValueError("Pattern not found in LLM response")

        print("LLM response format is incorrect, reprocessing...")
        llm_response = prompt_llm_to_correct_format(llm_response)
        response = llm_response  # Update response for reprocessing
        count += 1

    # If the response format is still incorrect after max retries
    logging.error(f"Failed to correct response format after {max_retries} retries: {response}")
    return None, None








def process_row(df, index, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords, predefined_context, predefined_reply, content_used, classes):
    row = df.iloc[index]
    video_title = row['video_title'] if 'video_title' in row and row['video_title'] else None
    transcript = row['transcriptions'] if 'transcriptions' in row and row['transcriptions'] else None
    comment = row['comment'] if 'comment' in row and row['comment'] else predefined_context
    reply = row['reply'] if 'reply' in row and row['reply'] else predefined_reply
    video_description = row['description'] if 'description' in row and row['description'] else None
    audio_event = row['audio_events'] if 'audio_events' in row and row['audio_events'] else None
    object = row['object'] if 'object' in row and row['object'] else None

    # Metadata
    view_count_video = row['viewCount_YT_video'] if 'viewCount_YT_video' in row and row['viewCount_YT_video'] else None
    like_count_video = row['likeCount_YT_video'] if 'likeCount_YT_video' in row and row['likeCount_YT_video'] else None
    favorite_count_video = row['favoriteCount_YT_video'] if 'favoriteCount_YT_video' in row and row['favoriteCount_YT_video'] else None
    comment_count_video = row['commentCount_YT_video'] if 'commentCount_YT_video' in row and row['commentCount_YT_video'] else None
    tags_video = row['tags_YT_video'] if 'tags_YT_video' in row and row['tags_YT_video'] else None
    channel_title_video = row['channelTitle_YT_video'] if 'channelTitle_YT_video' in row and row['channelTitle_YT_video'] else None
    author_reply = row['authorDisplayName_YT_reply'] if 'authorDisplayName_YT_reply' in row and row['authorDisplayName_YT_reply'] else None
    like_count_reply = row['likeCount_YT_reply'] if 'likeCount_YT_reply' in row and row['likeCount_YT_reply'] else None
    descriptionYT = row['description_YT_video'] if 'description_YT_video' in row and row['description_YT_video'] else None

    logging.info(f"Processing row: {index}")
    logging.debug(f"Video Title: {video_title}")
    logging.debug(f"Transcript: {transcript}")
    logging.debug(f"Video Description: {video_description}")
    logging.debug(f"Audio Events: {audio_event}")
    logging.debug(f"Object: {object}")
    logging.debug(f"Comment: {comment}")
    logging.debug(f"Reply: {reply}")
    logging.debug(f"View Count (Video): {view_count_video}")
    logging.debug(f"Like Count (Video): {like_count_video}")
    logging.debug(f"Favorite Count (Video): {favorite_count_video}")
    logging.debug(f"Comment Count (Video): {comment_count_video}")
    logging.debug(f"Tags (Video): {tags_video}")
    logging.debug(f"Channel Title (Video): {channel_title_video}")
    logging.debug(f"Author Name (Reply): {author_reply}")
    logging.debug(f"Like Count (Reply): {like_count_reply}")
    logging.debug(f"Video Description from YT: {descriptionYT}")

    print("BEFORE")
    print(f'Reply: {reply}')
    print(f'Comment: {comment}')
    print("AFTER")

    # Check if comment and reply are None before calling process_comment
    print(f'Comment before processing: {comment}')
    print(f'Reply before processing: {reply}')

    # Process comment using the process_comment function
    classification, explanation, comment, content_used, reply = process_comment(
        video_title, transcript, video_description, descriptionYT, audio_event, object, 
        view_count_video, like_count_video, favorite_count_video, comment_count_video, 
        tags_video, channel_title_video, author_reply, like_count_reply, 
        comment, reply, custom_entities, language, use_wikipedia_directly, 
        use_attention_score, use_spacy, use_propn, use_keywords, content_used, classes
    )

    print(f'Comment after processing: {comment}')
    print(f'Reply after processing: {reply}')

    return {
        'index': index,
        'comment': comment,
        'reply': reply,
        'classification': classification,
        'explanation': explanation,
        'content_used': content_used
    }



# Extract specified headers into separate variables
def get_row_of_dataset(df, index):
    #TODO add title to ALYT 
    video_title = df.iloc[index]['video_title']
    #video_url = df.iloc[index]['video_url']
    #region = df.iloc[index]['region']
    #language = df.iloc[index]['language']
    comment = df.iloc[index]['comment']
    transcription = df.iloc[index]['transcriptions']
    video_description = df.iloc[index]['description']
    
    # print("Video Titles:\n", video_title)
    # print("Video URLs:\n", video_url)
    # print("Regions:\n", region)
    # print("Languages:\n", language)
    # print("Transcriptions:\n", transcription)
    return video_title, comment, transcription, video_description


import logging
import requests
from spacy.lang.de.stop_words import STOP_WORDS as GERMAN_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS as ENGLISH_STOP_WORDS
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the BERT NER pipeline
#bert_ner_pipeline = pipeline("ner", model="dbmdz/bert-base-german-cased", tokenizer="dbmdz/bert-base-german-cased", aggregation_strategy="simple")

def extract_entities_with_bert(text):
    """Extract entities using a BERT model."""
    ner_results = bert_ner_pipeline(text)
    return [(result['word'], result['entity_group']) for result in ner_results]

# def extract_entities_with_spacy(doc):
#     """Extract entities using a SpaCy model."""
#     all_entity_types = {
#         "PERSON", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE",
#         "DATE", "TIME", "PERCENT", "MONEY", "NORP", "FAC", "GPE", "LAW", "QUANTITY", "ORDINAL", "CARDINAL"
#     }
#     return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in all_entity_types]

def extract_entities_with_spacy(doc):
    """Extract entities using a SpaCy model."""
    all_entity_types = {
        "PERSON", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE",
         "NORP", "FAC", "GPE", "LAW"
    }
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in all_entity_types]


def process_entities_from_comment(video_title, comment, reply, transcript, custom_keywords, language, use_wikipedia_directly=False, use_attention_score=False, use_spacy=True, use_propn=False, use_keywords=False):
    """Extract entities from the comment using dynamically loaded spaCy models or BERT transformer based on the flag."""

    #TODO: rethink this combining
    combined_text = f"{video_title if video_title else ''} {comment if video_title else ''} {reply if reply else ''} {transcript if transcript else ''}"
    #print("combined_text: ", combined_text)
    logging.info(f'Processing entities from comment: {combined_text}')

    if language =="DE":
        stopwords = GERMAN_STOP_WORDS.union(ENGLISH_STOP_WORDS)
    elif language =="EN":
        stopwords = ENGLISH_STOP_WORDS

    entities_with_types = []

    if use_spacy:
        nlp = get_spacy_model(combined_text, language)  # Load the spaCy model based on language detection
        doc = nlp(combined_text)
        spacy_entities = extract_entities_with_spacy(doc)
        logging.debug(f'spacy_entities: {spacy_entities}')
        entities_with_types.extend(spacy_entities)
    # else: #TODO:implement BERT entities with english model 
    #     bert_entities = extract_entities_with_bert(combined_text)
    #     logging.debug(f'bert_entities: {bert_entities}')
    #     entities_with_types.extend(bert_entities)
    #     #print("BERT entities: ", entities_with_types)

    patterns = [compile_keyword_pattern(kw) for kw in custom_keywords]
    custom_entities = []
    for pattern in patterns:
        custom_entities.extend([(match.group(), "CUSTOM") for match in pattern.finditer(combined_text)])

    #print("custom_entities: ", custom_entities)

    entities_with_types.extend(custom_entities)

    # Filter out stopwords from entities
    logging.debug(f'stopwords: {stopwords}')
    entities_with_types = [(entity, entity_type) for entity, entity_type in entities_with_types if entity.lower() not in stopwords]

    if use_spacy and not entities_with_types:
        # Fallback to multilingual model if no entities found and using spaCy
        if language == "EN": 
            nlp_fallback = spacy.load("en_core_web_sm") if nlp.meta['lang'] == 'en' else spacy.load("xx_ent_wiki_sm")
        elif language == "DE": 
            nlp_fallback = spacy.load("de_core_news_sm") if nlp.meta['lang'] == 'de' else spacy.load("xx_ent_wiki_sm")
        doc = nlp_fallback(combined_text)
        fallback_entities = extract_entities_with_spacy(doc)
        entities_with_types.extend(fallback_entities)
        entities_with_types = [(entity, entity_type) for entity, entity_type in entities_with_types if entity.lower() not in stopwords]

    if use_attention_score:
        entity_scores = score_entities([entity for entity, _ in entities_with_types], comment)
    else:
        entity_scores = None  # No scoring

    # Initialize dictionaries and lists for collecting content
    collected_content = {}
    collected_context = []

    print("FOUND ENTITIES:", entities_with_types)
    logging.info(f'FOUND ENTITIES: {entities_with_types}')

    # Function to check if a string is a pure number
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # Remove pure numbers (but do not remove single-character entities)
    filtered_entities_with_types = [(entity, entity_type) for entity, entity_type in entities_with_types if not is_number(entity)]
        
    print("FOUND FILTERED ENTITIES:", filtered_entities_with_types)
    logging.info(f'FOUND FILTERED ENTITIES: {filtered_entities_with_types}')
    
    # Remove duplicates while preserving order
    seen = set()
    unique_filtered_entities_with_types = []
    for entity, entity_type in filtered_entities_with_types:
        if (entity, entity_type) not in seen:
            seen.add((entity, entity_type))
            unique_filtered_entities_with_types.append((entity, entity_type))

    print("FOUND UNIQUE FILTERED ENTITIES:", unique_filtered_entities_with_types)
    logging.info(f'FOUND UNIQUE FILTERED ENTITIES: {unique_filtered_entities_with_types}')

    for entity, entity_type in unique_filtered_entities_with_types:
        content = get_wikipedia_content(entity, entity_type, use_wikipedia_directly, combined_text, language) if entities_with_types else "No content available"
        if content == "No content available" and use_spacy:
            lemmatized_entity = ' '.join([token.lemma_ for token in nlp(entity)])
            content = get_wikipedia_content(lemmatized_entity, entity_type, use_wikipedia_directly, combined_text, language)

        if content != "No content available":
            if entity_scores:
                ratio = entity_scores.get(entity, 1)
                max_length = int(ratio * 10000)
            else:
                max_length = 10000
            truncated_content = content[:max_length]
            collected_content[entity] = truncated_content
            collected_context.append(f"'{entity}': '{truncated_content}'")

        if content != "No content available":
            print(f'Content available for entity in processing function "{entity}"')
            process_and_store(content, entity)

    context = " ".join(collected_context)
    return context

def retrieve_context_for_comment(comment):
    """
    Retrieve relevant context for a comment directly using the GPT embeddings and the vector database retriever.
    """
    print("Retrieving context for classification...")

    # Assuming 'gpt4all_embd' is your GPT embedding model
    query_vector = gpt4all_embd.embed_query(comment)  # Get the embedding for the input comment

    # Set up and use the retriever with top k results
    retriever = db.as_retriever(search_kwargs={"k": 10})  # Adjust 'k' as needed

    # Fetch the top relevant documents based on the query
    retrieved_docs = retriever.invoke(comment)  # If necessary, adjust to use the query_vector directly

    # Process and return the documents. Adapt this according to your object's structure
    context_documents = [doc.page_content for doc in retrieved_docs]  # Adjusted to access the 'page_content' attribute
    return context_documents

def query_wikidata(entity, entity_type, language):
    """Query Wikidata for Wikipedia titles related to the provided term, specifically for German Wikipedia articles."""
    type_filter = ""
    if entity_type in {"PERSON", "PER"}:
        type_filter = "?item wdt:P31 wd:Q5."  # Instance of human
    elif entity_type == "ORG":
        type_filter = "?item wdt:P31 wd:Q43229."  # Instance of organization
    elif entity_type in {"LOC", "GPE"}:
        type_filter = "?item wdt:P31 wd:Q515."  # Instance of city or town

    if language == "DE":
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
            sparql.setQuery(f"""
                SELECT ?item ?itemLabel ?article WHERE {{
                    ?item ?label "{entity}"@de.
                    OPTIONAL {{
                        ?article schema:about ?item;
                        schema:isPartOf <https://de.wikipedia.org/>.
                    }}
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "de". }}
                }}
            """)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

    elif language == "EN":
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setQuery(f"""
            SELECT ?item ?itemLabel ?article WHERE {{
                ?item ?label "{entity}"@en.
                OPTIONAL {{
                    ?article schema:about ?item;
                    schema:isPartOf <https://en.wikipedia.org/>.
                }}
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    return [(result['item']['value'], result['itemLabel']['value'], result.get('article', {}).get('value')) for result in results["results"]["bindings"]]


# def get_wikipedia_content_from_title_or_entity(title, language):
#     """Fetch text content from Wikipedia for the given title or entity and return it, first trying German Wikipedia and falling back to English Wikipedia."""
#     def fetch_wikipedia_content(base_url, title):
#         params = {
#             "action": "query",
#             "format": "json",
#             "titles": title,
#             "prop": "extracts",
#             "explaintext": True,
#             "redirects": 1  # Follow redirects
#         }
#         response = requests.get(base_url, params=params).json()
#         page = next(iter(response['query']['pages'].values()), {})
#         if page.get("pageid"):
#             return page.get("extract", "No content available")
#         #print(f"No content available for '{title}'")
#         return "No content available"
    
#     # Try German Wikipedia first
#     content = fetch_wikipedia_content("https://de.wikipedia.org/w/api.php", title)
    
#     # If no content available, fall back to English Wikipedia
#     if content == "No content available" or content == '':
#         content = fetch_wikipedia_content("https://en.wikipedia.org/w/api.php", title)
#     #print(f"Content for '{title}' is: {content}")
#     return content



def get_wikipedia_content_from_title_or_entity(entity, language):
    #print("YOU ARE IN THE GET_WIKIPEDIA_CONTENT FUNCTION")
    
    """Fetch text content from Wikipedia for the given title or entity and return it, first trying German Wikipedia and falling back to English Wikipedia."""
    def fetch_wikipedia_content(base_url, title):
        params = {
            "action": "query",
            "format": "json",
            "titles": entity,
            "prop": "extracts",
            "explaintext": True,
            "redirects": 1  # Follow redirects
        }
        response = requests.get(base_url, params=params).json()
        page = next(iter(response['query']['pages'].values()), {})
        if page.get("pageid"):
            return page.get("extract", "No content available")
        #print(f"No content available for '{entity}'")
        return "No content available"

    if language == "DE":
        # Try German Wikipedia first
        content = fetch_wikipedia_content("https://de.wikipedia.org/w/api.php", entity)
        
        # If no content available, fall back to English Wikipedia
        if content == "No content available" or content == '':
            content = fetch_wikipedia_content("https://en.wikipedia.org/w/api.php", entity)
        #print(f"Content for '{title}' is: {content}")

    if language =="EN":
        # Try English Wikipedia first
        content = fetch_wikipedia_content("https://en.wikipedia.org/w/api.php", entity)
        
        # If no content available, fall back to English Wikipedia
        if content == "No content available" or content == '':
            content = fetch_wikipedia_content("https://de.wikipedia.org/w/api.php", entity)
        #print(f"Content for '{title}' is: {content}")
    
    return content

def get_wikipedia_content(entity, entity_type, use_wikipedia_directly, original_text, language):
    """Fetch text content from Wikipedia/Wikidata for the given entity"""
    if not use_wikipedia_directly:
        wikidata_results = query_wikidata(entity, entity_type, language)

        if wikidata_results:
            for item, label, article in wikidata_results:
                if article:
                    #print("wikidata_results article: ", article)

                    # Extract the title from the article URL which is the last part after '/'
                    title = article.split('/')[-1]
                    #print("wikidata title: ", title)
                    extracted_content = get_wikipedia_content_from_title_or_entity(title, language)

                    # Validate relevance
                    if validate_relevance(original_text, extracted_content) > 0.5:  # Threshold can be adjusted
                        return extracted_content
        print("No relevant content on wikipedia from wikidata title available")
        return "No content available"

    # Fall back to directly using Wikipedia if no Wikidata results
    print("Check for entity on wikipedia directly")
    return get_wikipedia_content_from_title_or_entity(entity, language)



def main_pipeline(df, classes, content_used, custom_entities, language, index=None, use_wikipedia_directly=False, use_attention_score=True, use_spacy=True, use_propn=True, use_keywords=False, predefined_context=None, predefined_reply=None):
    """Pipeline function to process either a specific row or all rows in the DataFrame."""
    
    set_logging()
  
    responses = []  # Initialize an empty list to accumulate responses
    
    if index is not None:
        print("index: ", index)
        response = process_row(df, index, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords, predefined_context, predefined_reply, content_used, classes)
        responses.append(response)
    else:
        for idx in range(len(df)):
            logging.info(f"\nProcessing row {idx}:\n")

            print(f"\nProcessing row {idx}:\n")

            # row = df.iloc[idx]

            #print("ROW: ", row)
            response = process_row(df, idx, custom_entities, language, use_wikipedia_directly, use_attention_score, use_spacy, use_propn, use_keywords, predefined_context, predefined_reply, content_used, classes)
            responses.append(response)
            time.sleep(seconds_per_request)  # Respect rate limits
    print("RESPONSE: ", response)


    # Convert the accumulated responses to a DataFrame
    response_df = pd.DataFrame(responses)
    #response_df = post_process_classifications(response_df, classes)
    response_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

    print("Processing complete. Output saved to", output_file)
    return response_df

