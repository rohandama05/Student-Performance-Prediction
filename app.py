from flask import Flask, request, render_template, send_file
import pickle
import pandas as pd
from fpdf import FPDF
import os
import sqlite3
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)

# LOAD MODEL
model = pickle.load(open("model.pkl", "rb"))

# DATABASE
def init_db():
    conn = sqlite3.connect('students.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (attendance INT, study_time INT, assignments INT, marks INT, result TEXT)''')
    conn.commit()
    conn.close()

init_db()

# HOME
@app.route('/')
def home():
    return render_template("index.html")

# PREDICT
@app.route('/predict', methods=['POST'])
def predict():

    attendance = int(request.form['attendance'])
    study_time = int(request.form['study_time'])
    assignments = int(request.form['assignments'])
    marks = int(request.form['marks'])

    # Convert marks (0–100 → 0–20)
    uci_marks = marks / 5

    data = pd.DataFrame([[attendance, study_time, assignments, uci_marks]],
                        columns=['attendance', 'study_time', 'assignments', 'previous_marks'])

    # Model prediction
    proba = model.predict_proba(data)[0]
    fail_prob = round(proba[0]*100, 2)
    pass_prob = round(proba[1]*100, 2)

    # Decision
    if marks < 40:
        result_label = "FAIL"
    else:
        result_label = "PASS"

    # GRADE SYSTEM
    if marks >= 85:
        grade = "A"
    elif marks >= 70:
        grade = "B"
    elif marks >= 50:
        grade = "C"
    elif marks >= 40:
        grade = "D"
    else:
        grade = "F"

    # Final output
    output = f"{result_label} | Marks: {marks} | Grade: {grade} | Pass: {pass_prob}% | Fail: {fail_prob}%"

    # SAVE TO DATABASE
    conn = sqlite3.connect('students.db')
    c = conn.cursor()
    c.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?)",
              (attendance, study_time, assignments, marks, output))
    conn.commit()
    conn.close()

    # CREATE STATIC FOLDER
    if not os.path.exists("static"):
        os.makedirs("static")

    # INPUT GRAPH
    plt.figure()
    plt.bar(['Attendance', 'Study Time', 'Assignments', 'Marks'],
            [attendance, study_time, assignments, marks])
    plt.title("Student Input")
    plt.savefig("static/input_graph.png")
    plt.close()

    # PDF REPORT
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10, txt="Student Performance Report", ln=True)
    pdf.cell(200, 10, txt=f"Attendance: {attendance}", ln=True)
    pdf.cell(200, 10, txt=f"Study Time: {study_time}", ln=True)
    pdf.cell(200, 10, txt=f"Assignments: {assignments}", ln=True)
    pdf.cell(200, 10, txt=f"Marks: {marks}", ln=True)
    pdf.cell(200, 10, txt=f"Result: {result_label}", ln=True)
    pdf.cell(200, 10, txt=f"Grade: {grade}", ln=True)
    pdf.cell(200, 10, txt=f"Pass Probability: {pass_prob}%", ln=True)
    pdf.cell(200, 10, txt=f"Fail Probability: {fail_prob}%", ln=True)

    pdf.output("report.pdf")

    return render_template("result.html", prediction=output)

# DOWNLOAD
@app.route('/download')
def download():
    return send_file("report.pdf", as_attachment=True)

# DASHBOARD
@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('students.db')
    c = conn.cursor()

    c.execute("SELECT * FROM results ORDER BY rowid DESC LIMIT 10")
    data = c.fetchall()
    conn.close()

    pass_count = sum(1 for row in data if "PASS" in row[4])
    fail_count = sum(1 for row in data if "FAIL" in row[4])

    # PIE CHART
    plt.figure()
    plt.pie([pass_count, fail_count], labels=['Pass', 'Fail'], autopct='%1.1f%%')
    plt.title("Pass vs Fail Distribution")
    plt.savefig("static/dashboard.png")
    plt.close()

    return render_template("dashboard.html",
                           data=data,
                           pass_count=pass_count,
                           fail_count=fail_count)

# RUN
if __name__ == "__main__":
    app.run(debug=True)