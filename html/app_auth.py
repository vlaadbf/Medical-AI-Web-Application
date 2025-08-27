from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
import bcrypt
import os
from datetime import datetime
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from tensorflow.keras.models import load_model


model = load_model("mobilenet_lung_model.h5")
class_names = ["Benign", "Malignant"]

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Conectare la MongoDB local
client = MongoClient("mongodb://localhost:27017")
db = client["dizertatie"]
users_collection = db["users"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Date generale
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]
        first_name = request.form["first_name"]
        last_name = request.form["last_name"]
        sex = request.form["sex"]
        birthdate = request.form["birthdate"]
        judet = request.form["judet"]

        # Spital și fumător (condiționat)
        spital = None
        smoker = None
        smoker_since = None
        smoker_years = None

        if role == "doctor":
            spital = request.form.get("spital_doctor")
        elif role == "pacient":
            spital = request.form.get("spital_pacient")
            smoker = request.form.get("smoker") == "true"
            smoker_since = request.form.get("smoker_since")
            if smoker and smoker_since:
                try:
                    smoker_years = datetime.today().year - int(smoker_since)
                except:
                    smoker_years = None

        # Verificare dacă utilizatorul există
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            return "Acest utilizator există deja."

        hashpass = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

        user_data = {
            "email": email,
            "password": hashpass,
            "role": role,
            "first_name": first_name,
            "last_name": last_name,
            "sex": sex,
            "birthdate": birthdate,
            "judet": judet,
            "spital": spital,
            "created_at": datetime.today().strftime("%Y-%m-%d")
        }

        if role == "pacient":
            user_data["smoker"] = smoker
            user_data["smoker_since"] = smoker_since
            user_data["smoker_years"] = smoker_years
            user_data["approved"] = False
            user_data["requested_doctor"] = None
            user_data["pending_images"] = []
            user_data["test_history"] = []

        if role == "doctor":
            user_data["patients"] = []

        users_collection.insert_one(user_data)
        return redirect(url_for("login"))
    
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = users_collection.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
            session["email"] = email
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        return "Login greșit."
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "email" not in session:
        return redirect(url_for("login"))
    user = users_collection.find_one({"email": session["email"]})
    doctor = None
    if user.get("approved") and user.get("requested_doctor"):
        doctor = users_collection.find_one({"email": user["requested_doctor"]})
    return render_template("dashboard.html", role=session["role"], user=user, doctor=doctor)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "email" not in session or session["role"] != "pacient":
        return redirect(url_for("login"))

    user = users_collection.find_one({"email": session["email"]})
    doctor = users_collection.find_one({"email": user["requested_doctor"]}) if user.get("approved") and user.get("requested_doctor") else None

    if request.method == "POST":
        files = request.files.getlist("images")
        saved_files = []
        for file in files:
            if file.filename:
                filename = f"{session['email'].replace('@', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
                path = os.path.join("static/uploads", filename)
                file.save(path)
                saved_files.append(filename)

        if saved_files:
            users_collection.update_one(
                {"email": session["email"]},
                {"$push": {"pending_images": {"$each": saved_files}},
                 "$set": {"upload_target": doctor["email"] if doctor else None}}
            )

        return redirect(url_for("dashboard"))

    return render_template("upload.html", user=user, role=session["role"], doctor=doctor)

@app.route("/cerere", methods=["GET", "POST"])
def cerere():
    if "email" not in session or session["role"] != "pacient":
        return redirect(url_for("login"))

    user = users_collection.find_one({"email": session["email"]})
    doctor = users_collection.find_one({"email": user.get("requested_doctor")}) if user.get("requested_doctor") else None

    if user.get("approved") and doctor:
        return render_template("cerere.html", approved=True, doctor=doctor, user=user, role=session["role"], doctors=None)

    doctors = users_collection.find({
        "role": "doctor",
        "email": {"$ne": user.get("requested_doctor")}
    })

    if request.method == "POST":
        selected_doctor = request.form["doctor"]
        users_collection.update_one(
            {"email": session["email"]},
            {"$set": {"requested_doctor": selected_doctor, "approved": False}}
        )
        doctor_info = users_collection.find_one({"email": selected_doctor})
        return render_template("confirmare_cerere.html", doctor=doctor_info, data=datetime.today().strftime("%Y-%m-%d"), user=user, role=session["role"], doctor_info=doctor_info)

    return render_template("cerere.html", doctors=doctors, user=user, role=session["role"], doctor=doctor, approved=False)


@app.route("/cereri_pacienti")
def cereri_pacienti():
    if "email" not in session or session["role"] != "doctor":
        return redirect(url_for("login"))

    user = users_collection.find_one({"email": session["email"]})
    cereri = users_collection.find({
        "role": "pacient",
        "requested_doctor": user["email"],
        "approved": False
    })

    return render_template("cereri_pacienti.html", cereri=cereri, user=user, role=session["role"], doctor=None)

@app.route("/aproba/<pacient_email>")
def aproba(pacient_email):
    if "email" not in session or session["role"] != "doctor":
        return redirect(url_for("login"))

    users_collection.update_one(
        {"email": pacient_email},
        {"$set": {"approved": True}}
    )

    users_collection.update_one(
        {"email": session["email"]},
        {"$addToSet": {"patients": pacient_email}}
    )

    return redirect(url_for("cereri_pacienti"))

from flask import request
from math import ceil
import unicodedata

def normalize(text):
    """Elimină diacriticele și transformă textul în litere mici"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()
@app.route("/pacientii_mei")
def pacientii_mei():
    if "email" not in session or session["role"] != "doctor":
        return redirect(url_for("login"))

    doctor = users_collection.find_one({"email": session["email"]})
    pacienti_raw = users_collection.find({"email": {"$in": doctor.get("patients", [])}})
    search = request.args.get("search", "").strip()
    page = request.args.get("page", 1, type=int)
    per_page = 10

    pacienti = []
    for p in pacienti_raw:
        birth_str = p.get("birthdate")
        try:
            birth = datetime.strptime(birth_str, "%Y-%m-%d")
            varsta = (datetime.today() - birth).days // 365
        except:
            varsta = "-"
        pacient = {
            "email": p.get("email", ""),
            "full_name": f"{p.get('first_name', '')} {p.get('last_name', '')}",
            "judet": p.get("judet", ""),
            "sex": p.get("sex", ""),
            "smoker": p.get("smoker", False),
            "birthdate": birth_str,
            "varsta": varsta,
            "pending": len(p.get("pending_images", [])),
            "istoric": len(p.get("test_history", [])),
        }

        # Filtru după căutare
        if search:
            text = normalize(pacient["full_name"] + " " + pacient["email"])
            if normalize(search) not in text:
                continue

        pacienti.append(pacient)

    # Paginate filtered results
    total = len(pacienti)
    total_pages = (total + per_page - 1) // per_page
    start = (page - 1) * per_page
    pacienti_pagina = pacienti[start:start+per_page]

    return render_template(
        "pacientii_mei.html",
        pacienti=pacienti_pagina,
        pagina=page,
        total_pagini=total_pages,
        search=search,
        user=doctor,
        role=session["role"],
        doctor=None
    )


@app.route("/detalii_pacient/<email>", methods=["GET", "POST"])
def detalii_pacient(email):
    if "email" not in session or session["role"] != "doctor":
        return redirect(url_for("login"))

    doctor = users_collection.find_one({"email": session["email"]})
    pacient = users_collection.find_one({"email": email})

    predictions = []
    mean_prediction = None

    if request.args.get("predict") == "1" and pacient.get("pending_images"):
        scores = []
        for fname in pacient["pending_images"]:
            path = os.path.join("static/uploads", fname)
            try:
                img = Image.open(path).convert("RGB").resize((224, 224))
                arr = np.array(img)
                arr = preprocess_input(arr)
                arr = np.expand_dims(arr, axis=0)
                pred = model.predict(arr)[0]
                label = class_names[np.argmax(pred)]
                confidence = float(np.max(pred))
                scores.append(pred)
                predictions.append({
                    "img": fname,
                    "label": label,
                    "conf": f"{confidence*100:.2f}%"
                })
            except Exception as e:
                predictions.append({"img": fname, "label": "Eroare", "conf": str(e)})

        if scores:
            mean_pred = np.mean(scores, axis=0)
            label = class_names[np.argmax(mean_pred)]
            score = f"{np.max(mean_pred)*100:.2f}%"
            mean_prediction = {"label": label, "score": score}

    if request.method == "POST":
        mesaj = request.form.get("mesaj")
        if mesaj and mean_prediction:
            result = {
                "date": datetime.today().strftime("%Y-%m-%d"),
                "message": mesaj,
                "from": doctor["email"],
                "summary": mean_prediction,
                "images": pacient.get("pending_images", [])
            }
            users_collection.update_one(
                {"email": email},
                {
                    "$set": {"last_result": result, "pending_images": []},
                    "$push": {"test_history": result}
                }
            )
            return redirect(url_for("pacientii_mei"))

    return render_template("detalii_pacient.html", pacient=pacient, user=doctor, role=session["role"], doctor=doctor, predictions=predictions, mean_prediction=mean_prediction)

@app.route("/rezultate")
def rezultate():
    if "email" not in session or session["role"] != "pacient":
        return redirect(url_for("login"))

    user = users_collection.find_one({"email": session["email"]})
    doctor = users_collection.find_one({"email": user["requested_doctor"]}) if user.get("approved") and user.get("requested_doctor") else None
    result = user.get("last_result")

    return render_template("rezultate.html", user=user, role="pacient", doctor=doctor, result=result)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
