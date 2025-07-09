import os
import pandas as pd

# Define new example mappings
category_map = {
    "Groceries": [
        "Safeway grocery", "Trader Joe's", "Walmart essentials", "Apna Bazaar",
        "Whole Foods vegetables", "TJ's snacks", "Kroger supplies"
    ],
    "Eating Out": [
        "Pizza at Domino's", "McDonald's lunch", "Dinner at Olive Garden",
        "Starbucks coffee", "Lunch at Chipotle", "Breakfast at IHOP"
    ],
    "Alcohol": [
        "Wine night", "Beer from Safeway", "Liquor store purchase",
        "Whiskey for party", "Chambong shots"
    ],
    "Shopping": [
        "Amazon order", "Target haul", "Costco shopping", "Myntra clothes",
        "H&M jeans", "Zara purchase", "Concert ticket", "Spotify plan"
    ],
    "Rent": [
        "Monthly Rent May", "Rent payment April", "Room rent",
        "Lease deposit", "Rent + utilities"
    ],
    "Utilities": [
        "Electricity bill", "Internet - Xfinity", "Water bill", "Mobile plan September",
        "Elec: April 1-May 8", "Gas charges", "Nov Dec electricity", "Mint mobile plan"
    ],
    "Travel": [
        "Flight to Mumbai", "Bus Madgaon to Thivim", "Airbnb Goa", "Trip to Portland",
        "Train to LA", "Hotel booking", "Uber to airport", "Lyft ride downtown",
        "Bus to Fremont", "Parking Sinquerim", "Train ticket Seattle", "Avis car rental"
    ],
    "Other": [
        "CVS pharmacy meds", "Therapist session", "Dental cleaning",
        "Emergency visit", "Doctor appointment", "Netflix subscription",
        "Movie night", "Amusement park ticket", "Bowling with friends"
    ]
}

# Create the directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Convert to DataFrame
rows = [{"category": cat, "example": ex} for cat, ex_list in category_map.items() for ex in ex_list]
df = pd.DataFrame(rows)

# Save to CSV
df.to_csv("data/category_examples.csv", index=False)
print("✅ category_examples.csv successfully saved.")
