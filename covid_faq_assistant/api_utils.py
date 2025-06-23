import httpx

# URL and API_KEY should be set according to the actual Dial API details provided
DIAL_API_URL = "https://ai-proxy.lab.epam.com/openai/models"
API_KEY = ""

def get_answer(questions, user_query):
    """
    Simulate sending a query to EPAM Dial API and receiving an enhanced answer.
    It processes results from VectorDB and sends a request to Dial API to get an enhanced response.

    Args:
    questions (list): List of potential questions closest to the user's query from VectorDB.
    user_query (str): The actual question asked by the user.

    Returns:
    str: The enhanced answer from the Dial API or a fallback response from the closest match.
    """
    if not questions:
        return "No relevant questions found. Try rephrasing your query."

    # For simplicity, we're assuming the first question is the best match
    best_match_question = questions[0]

    # Here we make a POST request to the Dial API - adapting this to the actual API requirements needed
    try:
        response = httpx.post(
            DIAL_API_URL,
            json={
                "question": best_match_question,
                "user_query": user_query
            },
            headers={'Authorization': f'Bearer {API_KEY}'}
        )
        response.raise_for_status()

        # Assuming that the API returns a JSON with an 'answer' field
        data = response.json()
        return data.get("answer", "No answer available at the moment.")

    except httpx.HTTPStatusError as e:
        print(f"Reque st to Dial API failed. Status Code: {e.response.status_code}. Return local best match.")
        return best_match_question  # Fallback response using the best match from the FAQ dataset
    
    except Exception as e:
        print(f"Error occurred: {e}. Return local best match.")
        return best_match_question  # Fallback response in case of other errors