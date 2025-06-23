import pickle

# Load the FAQ data
FAQ_DATA_PATH = "data/faq_data.pkl"

with open(FAQ_DATA_PATH, "rb") as f:
    faq_data = pickle.load(f)  # This might be a list of dictionaries or a DataFrame

# Ensure faq_data is a list of dictionaries
if isinstance(faq_data, dict):  # If it's a dictionary, convert it to a list
    faq_data = list(faq_data.values())

elif not isinstance(faq_data, list):  # If it's a Pandas DataFrame
    faq_data = faq_data.to_dict(orient="records")

search_term = "COVID-19"

print("\n🔍 Searching for related questions in FAQ dataset...\n")
found = False
for faq in faq_data:
    if search_term.lower() in str(faq.get("question", "")).lower():
        print(f"✅ Found: {faq['question']} -> {faq['answer']}")
        found = True

if not found:
    print("❌ No relevant FAQ found! You may need to add it manually.")




