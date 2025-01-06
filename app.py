from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

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

# Preprocess data
def preprocess_data():
    """Prepares the courses dataframe for recommendations with enhanced handling."""
    courses_df['Prerequisite'] = courses_df['Prerequisite'].fillna('None')
    courses_df['Combined_Info'] = courses_df['Course Name'] + ' ' + courses_df['Prerequisite']
    completed_courses = set(results_df['Course Code'].dropna())
    courses_df['Completed'] = courses_df['Course Code'].apply(lambda x: x in completed_courses)
    return courses_df

# Analyze academic performance
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

# TF-IDF-based recommendation system
def recommend_courses_with_tfidf(resume_text):
    """Recommend courses using TF-IDF for text vectorization."""
    # Vectorize the text
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    course_texts = courses_df['Combined_Info'].tolist()
    tfidf_matrix = tfidf_vectorizer.fit_transform(course_texts + [resume_text])
    
    # Compute cosine similarity
    user_vector = tfidf_matrix[-1]  # The last vector is the user's resume
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix[:-1]).flatten()
    
    # Add similarity scores to the dataframe
    courses_df['Similarity'] = similarity_scores
    
    # Filter and prioritize courses
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

# Preprocess data and analyze performance
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
        response = recommend_courses_with_tfidf(resume_text)

        return jsonify({
            "average_weighted_scores": avg_weighted_scores.to_dict(),
            "recommendations": response
        })

    except Exception as e:
        logging.error(f"Error in /recommend: {e}")
        return jsonify({"error": "An error occurred during processing."}), 500

if __name__ == "__main__":
    # Bind to the port assigned by Render
    port = int(os.environ.get("PORT", 9000))
    app.run(host="0.0.0.0", port=port)
