from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
from sentence_transformers import SentenceTransformer, util
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Initialize models as None for lazy loading
nlp = None
sbert_model = None

def get_spacy_model():
    """Dynamically load the spaCy model."""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        logging.info("spaCy model loaded successfully.")
    return nlp

def get_sbert_model():
    """Dynamically load the SBERT model."""
    global sbert_model
    if sbert_model is None:
        try:
            model_name = "all-MiniLM-L6-v2"
            model_cache_path = "./models/"
            if not os.path.exists(model_cache_path):
                os.makedirs(model_cache_path)
            sbert_model = SentenceTransformer(model_name, cache_folder=model_cache_path)
            logging.info(f"SBERT model '{model_name}' loaded successfully from {model_cache_path}.")
        except Exception as e:
            logging.error(f"Error loading SBERT model: {e}")
            raise
    return sbert_model

# Load data
try:
    CONFIG = {
        "results_file": "Final_Results_All_Semesters.csv",
        "courses_file": "Finalized_Course_Directory_with_Adjusted_Course_Names.csv",
        "similarity_threshold": 0.1,
        "default_recommendations": 5
    }

    results_df = pd.read_csv(CONFIG['results_file'])
    courses_df = pd.read_csv(CONFIG['courses_file'])
    logging.info("Data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

# Validate dataframes
def validate_dataframes():
    required_results_columns = {"Semester", "Course Name", "Course Code", "Course Type", "Credits", "Grade"}
    required_courses_columns = {"Course Name", "Course Code", "Prerequisite"}

    if not required_results_columns.issubset(results_df.columns):
        raise ValueError(f"Missing columns in results_df: {required_results_columns - set(results_df.columns)}")
    if not required_courses_columns.issubset(courses_df.columns):
        raise ValueError(f"Missing columns in courses_df: {required_courses_columns - set(courses_df.columns)}")

validate_dataframes()

# Helper Functions
def extract_resume_keywords(resume_text):
    """Extracts domain-relevant keywords from a resume using spaCy."""
    doc = get_spacy_model()(resume_text)
    keywords = []

    # Extract specific entities
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'SKILL', 'GPE', 'LANGUAGE', 'WORK_OF_ART', 'EVENT']:
            keywords.append(ent.text)

    # Include noun chunks (e.g., "data analysis", "machine learning")
    keywords.extend([chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1])

    # Deduplicate and join keywords
    return " ".join(set(keywords))

def analyze_academic_performance():
    """Analyzes academic performance to identify strong and weak areas."""
    grade_weights = {
        'A+': 4.5, 'A': 4.0, 'A-': 3.7,
        'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7,
        'D+': 1.3, 'D': 1.0, 'F': 0.0
    }
    results_df['Grade Weight'] = results_df['Grade'].map(grade_weights)
    results_df['Weighted Score'] = results_df['Grade Weight'] * pd.to_numeric(results_df['Credits'], errors='coerce')

    avg_weighted_scores = results_df.groupby('Course Type')['Weighted Score'].mean().sort_values(ascending=False)
    strong_courses = results_df[results_df['Grade Weight'] >= 3.0]['Course Code'].tolist()
    weak_courses = results_df[results_df['Grade Weight'] <= 2.0]['Course Code'].tolist()

    return avg_weighted_scores, strong_courses, weak_courses

def preprocess_data():
    """Prepares the courses dataframe for recommendations with enhanced handling."""
    # Fill missing prerequisites with a placeholder
    courses_df['Prerequisite'] = courses_df['Prerequisite'].fillna('None')

    # Combine relevant course information for matching
    courses_df['Combined_Info'] = courses_df['Course Name'] + ' ' + courses_df['Prerequisite']

    # Mark courses as completed if they appear in the results data
    completed_courses = set(results_df['Course Code'].dropna())
    courses_df['Completed'] = courses_df['Course Code'].apply(lambda x: x in completed_courses)

    return courses_df

def recommend_courses_with_sbert(resume_text):
    """Recommend courses using SBERT for semantic matching."""
    resume_keywords = extract_resume_keywords(resume_text)
    if not resume_keywords:
        resume_keywords = resume_text

    # Compute embeddings
    model = get_sbert_model()
    course_embeddings = model.encode(courses_df['Combined_Info'].tolist(), convert_to_tensor=True)
    user_embedding = model.encode([resume_keywords], convert_to_tensor=True)

    # Calculate similarities
    similarities = util.pytorch_cos_sim(user_embedding, course_embeddings)[0].cpu().numpy()

    # Add similarity scores to the dataframe
    courses_df['Similarity'] = similarities

    # Identify strong and weak courses first (based on academic performance)
    recommendations = courses_df[~courses_df['Completed']]

    recommendations['Priority'] = recommendations['Course Code'].apply(
        lambda x: 3 if x in strong_courses else (1 if x in weak_courses else 2)
    )

    # Sort by priority and similarity
    recommendations = recommendations.sort_values(by=['Priority', 'Similarity'], ascending=[False, False]).head(5)

    # Fallback to random suggestions if no strong matches
    if recommendations['Similarity'].max() < 0.1:
        recommendations = courses_df[~courses_df['Completed']].sample(5)

    return recommendations[['Course Code', 'Course Name', 'Prerequisite', 'Similarity', 'Priority']].to_dict(orient='records')

# Preprocess data
courses_df = preprocess_data()
avg_weighted_scores, strong_courses, weak_courses = analyze_academic_performance()

@app.route('/')
def home():
    """Serve the frontend page."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_courses():
    """Endpoint to recommend courses based on resume text."""
    try:
        data = request.get_json()
        if not data or 'resume_text' not in data:
            return jsonify({"error": "Missing 'resume_text' in request"}), 400

        resume_text = data['resume_text']
        response = recommend_courses_with_sbert(resume_text)

        return jsonify({
            "average_weighted_scores": avg_weighted_scores.to_dict(),
            "recommendations": response
        })

    except Exception as e:
        logging.error(f"Error in /recommend: {e}")
        return jsonify({"error": "An error occurred during processing."}), 500

if __name__ == "__main__":
    # Bind to the port assigned by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
