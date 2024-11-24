import os
import logging
import re
import requests
import spacy
import nltk
from flask import Flask, jsonify, request
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
MODEL_NAME = os.getenv("MODEL_NAME", "your-finetuned-portuguese-political-model")
MEANINGCLOUD_API_KEY = os.getenv("MEANINGCLOUD_API_KEY")
SPACY_MODEL = os.getenv("SPACY_MODEL", "pt_core_news_sm")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 512))
API_URL = "https://api.meaningcloud.com/sentiment-2.1"
EMAIL_REGEX = r'\S+@\S+'
URL_REGEX = r'http\S+|www.\S+'
PHONE_REGEX = r'\b\d{2,3}[-.\s]??\d{4,5}[-.\s]??\d{4}\b'

ERROR_LOADING_SPACY_MODEL_MSG = "Error loading SpaCy model: {}"
SPACY_MODEL_DOWNLOAD_INSTRUCTION = "Please run 'python -m spacy download {}' to download the model."

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


# Load SpaCy model
def load_spacy_model():
    try:
        spacy_nlp = spacy.load(SPACY_MODEL)
        logging.info("SpaCy model loaded successfully.")
        return spacy_nlp
    except OSError:
        logging.info(f"{SPACY_MODEL} not found. Downloading...")
        from subprocess import check_call
        check_call(["python", "-m", "spacy", "download", SPACY_MODEL])
        spacy_nlp = spacy.load(SPACY_MODEL)
        logging.info("SpaCy model downloaded and loaded successfully.")
        return spacy_nlp
    except Exception as e:
        logging.error(ERROR_LOADING_SPACY_MODEL_MSG.format(e))
        raise SystemExit(SPACY_MODEL_DOWNLOAD_INSTRUCTION.format(SPACY_MODEL))


# Load Hugging Face model and tokenizer
def load_huggingface_model_tokenizer():
    try:
        huggingface_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
        huggingface_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
        logging.info("Hugging Face model and tokenizer loaded successfully.")
        return huggingface_tokenizer, huggingface_model
    except Exception as e:
        raise RuntimeError(f"Error loading model/tokenizer: {e}")


# Load models
nlp = load_spacy_model()
tokenizer, model = load_huggingface_model_tokenizer()

# Flask application
app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Sentiment Analysis API"})


def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing personal info, anonymizing entities, and filtering stopwords.
    Args:
        text (str): Input text to preprocess.
    Returns:
        str: Processed text ready for model inference.
    """
    text = re.sub(EMAIL_REGEX, '[EMAIL]', text)
    text = re.sub(URL_REGEX, '[URL]', text)
    text = re.sub(PHONE_REGEX, '[PHONE]', text)
    tokens = word_tokenize(text, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token.isalpha()]
    doc = nlp(' '.join(filtered_tokens))
    return ' '.join(token.lemma_ for token in doc)


def analyze_political_view(text: str) -> dict:
    """
    Perform political view classification using the Hugging Face model.
    Args:
        text (str): Input text to analyze.
    Returns:
        dict: Analysis results including predicted label and probabilities.
    """
    processed_text = preprocess_text(text)
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
    outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=1).squeeze()
    predicted_label_idx = probabilities.argmax().item()
    predicted_label = model.config.id2label[predicted_label_idx]
    return {
        "original_text": text,
        "processed_text": processed_text,
        "predicted_label": predicted_label,
        "probabilities": {label: prob.item() for label, prob in zip(model.config.id2label.values(), probabilities)}
    }


def analyze_sentiment(text: str) -> dict:
    """
    Perform sentiment analysis using the MeaningCloud Sentiment Analysis API.
    Args:
        text (str): Input text to analyze.
    Returns:
        dict: JSON response from the MeaningCloud API.
    """
    if not MEANINGCLOUD_API_KEY:
        raise RuntimeError("MeaningCloud API key is not set. Please check your .env file.")
    payload = {
        "key": MEANINGCLOUD_API_KEY,
        "txt": text,
        "lang": "pt"
    }
    try:
        response = requests.post(API_URL, data=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error connecting to MeaningCloud API: {e}")


@app.route('/analyze/sentiment', methods=['POST'])
def analyze_sentiment_endpoint():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = analyze_sentiment(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Welcome to MPBoana!")
    while True:
        print("\nOptions:")
        print("1. Political View Classification")
        print("2. Sentiment Analysis")
        print("3. Quit")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            text = input("Enter text for political view classification: ").strip()
            if not text:
                print("No input provided.")
                continue
            try:
                result = analyze_political_view(text)
                print("\nPolitical View Analysis Result:")
                print(result)
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "2":
            text = input("Enter text for sentiment analysis: ").strip()
            if not text:
                print("No input provided.")
                continue
            try:
                result = analyze_sentiment(text)
                print("\nSentiment Analysis Result:")
                print(result)
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please choose again.")