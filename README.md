# Course Recommender System

## Overview
The **Course Recommender System** is a Flask-based web application that provides personalized course recommendations to students based on their academic performance and interests. It uses **TF-IDF vectorization** and **cosine similarity** to match user input (resume or preferences) with available courses.

## Features
- **Personalized Course Recommendations**: Suggests courses based on past academic performance and user input.
- **TF-IDF & Cosine Similarity**: Text-based matching of user interests with course descriptions.
- **Academic Performance Analysis**: Identifies strong and weak subjects based on past grades.
- **Dynamic Web Interface**: Built with **Bootstrap**, allowing users to input text and receive recommendations in real time.
- **Flask API**: Handles user queries and returns ranked course suggestions.

## How It Works
1. **User Input**: The user enters text (e.g., resume, interests).
2. **Preprocessing**:
   - Loads past academic records and course descriptions.
   - Identifies completed courses to avoid duplicate recommendations.
3. **Similarity Matching**:
   - Uses **TF-IDF vectorization** to convert text into numerical form.
   - Computes **cosine similarity** between user input and course descriptions.
4. **Prioritization**:
   - Assigns a **priority score** to each course based on past performance.
   - Strong subjects get higher priority; weak subjects get lower priority.
5. **Recommendation Generation**:
   - Sorts courses by priority and similarity score.
   - If no strong matches are found, suggests random courses.
6. **Frontend Display**:
   - Results are dynamically populated in a **Bootstrap-styled table**.

## Project Structure
```
Course-Recommender/
│── Final_Results_All_Semesters.csv          # Student academic history
│── Finalized_Course_Directory.csv           # Available courses
│── app.py                                   # Flask backend
│── requirements.txt                         # Dependencies
│── Procfile                                 # Deployment configuration
│── templates/
│   ├── index.html                           # Frontend interface
```

## Installation & Setup
### 1. Clone the Repository
```sh
git clone https://github.com/Sarvajeet2003/Course-Recommender.git
cd Course-Recommender
```

### 2. Install Dependencies
Ensure Python and pip are installed, then run:
```sh
pip install -r requirements.txt
```

### 3. Run the Flask Application
```sh
python app.py
```

The application will be accessible at `http://127.0.0.1:5000/`.

## API Endpoints
### `/` - Home Page
Serves the **index.html** frontend where users can input text and receive recommendations.

### `/recommend` - Course Recommendation API
**Method:** `POST`
**Payload:** `{ "resume_text": "User input text" }`
**Response:** JSON list of recommended courses with similarity scores and priorities.

## Deployment
### Deploy on Heroku
1. Login to Heroku and create an app:
   ```sh
   heroku login
   heroku create course-recommender
   ```
2. Push the code to Heroku:
   ```sh
   git push heroku main
   ```
3. Scale the app and open in browser:
   ```sh
   heroku ps:scale web=1
   heroku open
   ```

## Future Enhancements
- **User Authentication**: Personalized recommendations based on user profiles.
- **Improved NLP**: Use **BERT embeddings** instead of TF-IDF for better accuracy.
- **Database Integration**: Store and track user progress dynamically.
- **Visualization**: Graph-based insights on academic performance trends.

## License
This project is open-source under the **MIT License**.

## Author
Developed by [Sarvajeet2003](https://github.com/Sarvajeet2003).
