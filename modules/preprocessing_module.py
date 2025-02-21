import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

# Ensure required resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
nltk.download('vader_lexicon')
# Load spaCy English NLP model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text, use_stemming=False, use_lemmatization=True):
    """
    Cleans and preprocesses text by:
    - Lowercasing
    - Removing punctuation and numbers
    - Tokenizing
    - Removing stopwords
    - Applying stemming or lemmatization
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    elif use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens  # Return list of cleaned tokens


def generate_bow_ngrams(tokens_list, ngram_range=(1, 1)):
    """
    Converts a single list of tokens into:
    - Bag of Words (BoW)
    - N-grams (unigrams, bigrams, trigrams, etc.)
    """
    text = " ".join(tokens_list)

    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform([text])
    bow_dict = dict(zip(vectorizer.get_feature_names_out(), bow_matrix.toarray()[0]))

    def generate_ngrams(tokens, n):
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    ngrams_list = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngrams_list.extend(generate_ngrams(tokens_list, n))

    ngram_counts = Counter(ngrams_list)

    return bow_dict, ngram_counts


def preprocess_text(text, use_stemming=False, use_lemmatization=True):
    """
    Cleans and preprocesses text by:
    - Lowercasing
    - Removing punctuation and numbers
    - Tokenizing
    - Removing stopwords
    - Applying stemming or lemmatization

    Args:
        text (str): Input text
        use_stemming (bool): Apply stemming (default False)
        use_lemmatization (bool): Apply lemmatization (default True)

    Returns:
        list: Preprocessed tokens
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    elif use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def generate_bow_ngrams_batch(tokenized_texts, ngram_range=(1, 1)):
    """
    Generates Bag of Words and N-grams for a batch of tokenized texts.

    Args:
        tokenized_texts (list of list): List of tokenized sentences
        ngram_range (tuple): Range of n-grams to extract (default (1,1))

    Returns:
        tuple: (BoW dictionary, N-gram counts)
    """
    all_texts = [" ".join(tokens) for tokens in tokenized_texts]

    vectorizer = CountVectorizer(ngram_range=ngram_range)
    bow_matrix = vectorizer.fit_transform(all_texts)
    bow_dict = dict(zip(vectorizer.get_feature_names_out(), bow_matrix.toarray().sum(axis=0)))

    def generate_ngrams(tokens, n):
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    all_ngrams = []
    for tokens in tokenized_texts:
        for n in range(ngram_range[0], ngram_range[1] + 1):
            all_ngrams.extend(generate_ngrams(tokens, n))

    ngram_counts = Counter(all_ngrams)

    return bow_dict, ngram_counts

def generate_tfidf(tokenized_texts, ngram_range=(1, 1)):
    """
    Generates TF-IDF representation for tokenized texts.

    Args:
        tokenized_texts (list of list): List of tokenized texts
        ngram_range (tuple): Range of n-grams to include in TF-IDF

    Returns:
        dict: TF-IDF scores
    """
    all_texts = [" ".join(tokens) for tokens in tokenized_texts]
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    tfidf_dict = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray().sum(axis=0)))

    return tfidf_dict

def train_word2vec(tokenized_texts, vector_size=100, window=5, min_count=1):
    """
    Trains a Word2Vec model on tokenized texts.

    Args:
        tokenized_texts (list of list): List of tokenized texts
        vector_size (int): Size of word vectors
        window (int): Context window size
        min_count (int): Minimum word frequency

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model
    """
    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

def extract_named_entities(text):
    """
    Extracts named entities using SpaCy.

    Args:
        text (str): Input text

    Returns:
        dict: Named entities categorized by type
    """
    doc = nlp(text)
    entities = {ent.label_: [] for ent in doc.ents}
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return entities

def analyze_sentiment(text):
    """
    Analyzes sentiment using VADER.

    Args:
        text (str): Input text

    Returns:
        dict: Sentiment scores
    """
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def train_lda(tokenized_texts, num_topics=3):
    """
    Trains an LDA model for topic modeling.

    Args:
        tokenized_texts (list of list): List of tokenized texts
        num_topics (int): Number of topics

    Returns:
        tuple: (Trained LDA model, Dictionary)
    """
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    
    return lda_model, dictionary
