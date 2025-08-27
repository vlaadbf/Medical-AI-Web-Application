import json
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["dizertatie"]
collection = db["users"]

with open("users_sample.json", "r", encoding="utf-8") as f:
    users = json.load(f)

# Dacă e un singur utilizator, îl punem într-o listă
if isinstance(users, dict):
    users = [users]

# Convertim parolele în binar (pentru bcrypt dacă vrei să le compari ulterior)
for user in users:
    user["password"] = user["password"].encode("utf-8")

collection.insert_many(users)
print("✔️ Utilizatori importați fără dublă criptare.")
