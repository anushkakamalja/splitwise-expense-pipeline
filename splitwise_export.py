import os
import json
import webbrowser
import pandas as pd
from requests_oauthlib import OAuth1Session
import urllib.parse
from splitwise import Splitwise

# Load credentials
with open("credentials.json", "r") as f:
    creds = json.load(f)

CONSUMER_KEY = creds["consumer_key"]
CONSUMER_SECRET = creds["consumer_secret"]
TOKEN_FILE = "access_token.json"

REQUEST_TOKEN_URL = "https://secure.splitwise.com/api/v3.0/get_request_token"
AUTHORIZE_URL = "https://secure.splitwise.com/authorize"
ACCESS_TOKEN_URL = "https://secure.splitwise.com/api/v3.0/get_access_token"


def authenticate():
    print("🔐 Starting OAuth1 flow...")

    oauth = OAuth1Session(CONSUMER_KEY, client_secret=CONSUMER_SECRET)
    fetch_response = oauth.fetch_request_token(REQUEST_TOKEN_URL)

    if isinstance(fetch_response, str):
        fetch_response = dict(urllib.parse.parse_qsl(fetch_response))

    print("fetch_request_token response:", fetch_response)

    resource_owner_key = fetch_response.get("oauth_token")
    resource_owner_secret = fetch_response.get("oauth_token_secret")

    auth_url = f"{AUTHORIZE_URL}?oauth_token={resource_owner_key}"
    print("🌐 Opening browser to authorize...")
    webbrowser.open(auth_url)

    verifier = input("📥 Paste the oauth_verifier from the URL here: ").strip()

    oauth = OAuth1Session(
        CONSUMER_KEY,
        client_secret=CONSUMER_SECRET,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier,
    )
    access_token_response = oauth.fetch_access_token(ACCESS_TOKEN_URL)

    if isinstance(access_token_response, str):
        access_token_response = dict(urllib.parse.parse_qsl(access_token_response))

    access_token = {
        "oauth_token": access_token_response["oauth_token"],
        "oauth_token_secret": access_token_response["oauth_token_secret"]
    }

    with open(TOKEN_FILE, "w") as f:
        json.dump(access_token, f)

    print("✅ Access token obtained and saved.")
    
    sObj = Splitwise(CONSUMER_KEY, CONSUMER_SECRET)
    sObj.setAccessToken(access_token)
    return sObj


def load_authenticated_splitwise():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            token = json.load(f)
        sObj = Splitwise(CONSUMER_KEY, CONSUMER_SECRET)
        sObj.setAccessToken(token)
        return sObj
    else:
        return authenticate()

def fetch_and_save_expenses():
    sw = load_authenticated_splitwise()
    all_expenses = sw.getExpenses(limit=1000)

    expense_list = []
    for e in all_expenses:
        users = e.getUsers()
        if users and len(users) > 0:
            first_user = users[0]
            # Sometimes getUser() may be missing; try accessing user directly
            user_obj = first_user.getUser() if callable(getattr(first_user, "getUser", None)) else first_user
            paid_by = user_obj.getFirstName() if callable(getattr(user_obj, "getFirstName", None)) else "Unknown"
        else:
            paid_by = "Unknown"

        expense_list.append({
            "date": str(e.getDate()),
            "amount": e.getCost(),
            "currency": e.getCurrencyCode(),
            "paid_by": paid_by,
            "description": e.getDescription(),
        })

    df = pd.DataFrame(expense_list)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/expenses.csv", index=False)
    print("💾 Saved expenses to data/expenses.csv")


if __name__ == "__main__":
    fetch_and_save_expenses()
