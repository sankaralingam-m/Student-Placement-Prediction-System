import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
model = joblib.load("model/placement_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get form data
    student_name = request.form["student_name"]

    age = int(request.form["age"])
    gender = int(request.form["gender"])
    degree = int(request.form["degree"])
    branch = int(request.form["branch"])
    cgpa = float(request.form["cgpa"])
    internships = int(request.form["internships"])
    projects = int(request.form["projects"])
    coding_skills = int(request.form["coding_skills"])
    communication_skills = int(request.form["communication_skills"])
    aptitude = int(request.form["aptitude"])
    soft_skills = int(request.form["soft_skills"])
    certifications = int(request.form["certifications"])
    backlogs = int(request.form["backlogs"])

    # Create feature array
    features = np.array([[
        age,
        gender,
        degree,
        branch,
        cgpa,
        internships,
        projects,
        coding_skills,
        communication_skills,
        aptitude,
        soft_skills,
        certifications,
        backlogs
    ]])

    # Predict
    prediction = model.predict(features)

    # Confidence score
    probability = model.predict_proba(features)
    confidence = round(max(probability[0]) * 100, 2)

    # Result
    if prediction[0] == 1:

        status = "PLACED ✅"

        recommendation = """
        Excellent profile.
        Continue improving projects,
        certifications and practical skills.
        """

    else:

        status = "NOT PLACED ❌"

        recommendation = """
        Improve coding skills,
        aptitude score,
        projects and certifications.
        """

    return render_template(
        "result.html",
        student_name=student_name,
        status=status,
        confidence=confidence,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run(debug=True)