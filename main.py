from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = os.getenv("MODEL_NAME", "your-finetuned-portuguese-political-model")
MEANINGCLOUD_API_KEY = os.getenv("MEANINGCLOUD_API_KEY")
SPACY_MODEL = os.getenv("SPACY_MODEL", "pt_core_news_sm")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load Spacy model
def load_spacy_model(model_name):
    try:
        spacy_nlp = spacy.load(model_name)
        logging.info("SpaCy model loaded successfully.")
        return spacy_nlp
    except OSError:
        logging.info(f"{model_name} not found. Downloading...")
        from subprocess import check_call
        check_call(["python", "-m", "spacy", "download", model_name])
        spacy_nlp = spacy.load(model_name)
        logging.info("SpaCy model downloaded and loaded successfully.")
        return spacy_nlp
    except Exception as e:
        logging.error(f"Error loading SpaCy model: {e}")
        raise SystemExit(f"Please run 'python -m spacy download {model_name}' to download the model.")

# Load Hugging Face model and tokenizer
def load_huggingface_model_tokenizer(model_name, token):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
        logging.info