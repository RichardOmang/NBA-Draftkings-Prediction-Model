import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Step 1: Fetch the Data
@st.cache
def fetch_data():
    # Replace '2025' with actual season or dynamic value
    url = 'https://api.sportsdata.io/v3/nba/stats/json/PlayerSeasonStats/2025?key=550093bb07c7444eb010a108e523d99b'
    api_key = '550093bb07c7444eb010a108e523d99b'
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        print("Columns in the dataset:", df.columns)
        return df
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Step 2: Preprocess the Data
def preprocess_data(df):
    # Drop rows with missing values in relevant columns
    df.dropna(subset=['Points', 'Assists', 'Rebounds', 'FantasyPointsDraftKings', 'Minutes'], inplace=True)

    # Select relevant features for modeling
    features = ['Points', 'Assists', 'Rebounds', 'Minutes', 'Steals', 'BlockedShots']  # Customize this based on your data
    target = 'FantasyPointsDraftKings'
    
    # Check if all necessary columns exist
    if not all(col in df.columns for col in features + [target]):
        print("Missing columns for modeling")
        return None, None
    
    if df.empty:
        print("The dataset is empty after dropping rows with missing values.")
        return None, None
    
    X = df[features]  # Features
    y = df[target]    # Target variable

    # Apply Min-Max Scaling to the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
# Step 3: Train the Model
def train_model(X, y):
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

    return model

# Step 4: Deploy with Streamlit (optional)
def deploy_model(model):
    # Streamlit user interface
    st.title("NBA Player Projection Model")

    # Take input for a new player's stats
    points = st.number_input("Points", min_value=0, max_value=99)
    assists = st.number_input("Assists", min_value=0, max_value=99)
    rebounds = st.number_input("Rebounds", min_value=0, max_value=99)
    minutes = st.number_input("Minutes", min_value=0, max_value=99)
    steals = st.number_input("Steals", min_value=0, max_value=99)
    blockedshots = st.number_input("BlockedShots", min_value=0, max_value=99)

    # Create a DataFrame for the user input
    user_data = pd.DataFrame([[points, assists, rebounds, minutes, steals, blockedshots]],
                             columns=['Points', 'Assists', 'Rebounds', 'Minutes', 'Steals', 'BlockedShots'])

    # Get the prediction from the model
    if st.button("Get Projection"):
        projection = model.predict(user_data)
        st.write(f"Projected Fantasy Points: {projection[0]}")

# Main function to tie everything together
def main():
    # Fetch the data
    df = fetch_data()
    
    if df is not None:
        # Preprocess the data
        X, y = preprocess_data(df)

        if X is not None and y is not None:
            # Train the model
            model = train_model(X, y)

            # Deploy the model with Streamlit
            deploy_model(model)

# Run the main function
if __name__ == "__main__":
    main()