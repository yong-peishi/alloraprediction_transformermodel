from zipfile import ZipFile
import json
import os
import pickle
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dropout, Dense, Bidirectional
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files


def download_data(token, training_days, region, data_provider):
    if os.path.exists(binance_data_path):
        for filename in os.listdir(binance_data_path):
            filepath = os.path.join(binance_data_path, filename)
            if os.path.isfile(filepath):
                try:
                    print('Removing files')
                    os.remove(filepath)
                except:
                    pass
   
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")
    
def format_data(files, data_provider):
    if not files:
        print("Already up to date")
        return
    
    if data_provider == "binance":
        files = sorted([x for x in os.listdir(binance_data_path) if x.startswith(f"{TOKEN}USDT")])
    elif data_provider == "coingecko":
        files = sorted([x for x in os.listdir(coingecko_data_path) if x.endswith(".json")])

    # No files to process
    if len(files) == 0:
        return

    price_df = pd.DataFrame()
    if data_provider == "binance":
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file)

            if not zip_file_path.endswith(".zip"):
                continue

            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = [
                "start_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "end_time",
                "volume_usd",
                "n_trades",
                "taker_volume",
                "taker_volume_usd",
            ]
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)
    elif data_provider == "coingecko":
        for file in files:
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close"
                ]
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.drop(columns=["timestamp"], inplace=True)
                df.set_index("date", inplace=True)
                price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)


def load_frame(frame, timeframe):
    print(f"Loading data...")
    df = frame.loc[:,['open','high','low','close']].dropna()
    df[['open','high','low','close']] = df[['open','high','low','close']].apply(pd.to_numeric)
    df['date'] = frame['date'].apply(pd.to_datetime)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()

def create_lstm_data(df, target_col, n_steps, n_future_steps):
    x, y = [], []
    for i in range(n_steps, len(df) - n_future_steps):
        x.append(df.iloc[i-n_steps:i].values)
        y.append(df.iloc[i + n_future_steps][target_col])

    return np.array(x),np.array(y)

def train_model(timeframe):
    # Load the price data
    price_data = pd.read_csv(training_price_data_path)
    '''
    df = load_frame(price_data, timeframe)

    print(df.tail())

    y_train = df['close'].shift(-1).dropna().values
    X_train = df[:-1]

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")

    # Define the model
    if MODEL == "LinearRegression":
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif MODEL == "SVR":
        model = SVR()
        model.fit(X_train, y_train)
    elif MODEL == "KernelRidge":
        model = KernelRidge()
        model.fit(X_train, y_train)
    elif MODEL == "BayesianRidge":
        model = BayesianRidge()
        model.fit(X_train, y_train)
    # Add more models here
    else:
        raise ValueError("Unsupported model")
    
    # Train the model
    #model.fit(X_train, y_train)

    # create the model's parent directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # Save the trained model to a file
    if MODEL == 'LinearRegression' or MODEL =='SVR' or MODEL=='KernelRidge' or MODEL == 'BayesianRidge':
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
    '''
    if MODEL == 'Transformer':
        df = pd.DataFrame()

        # Convert 'date' to a numerical value (timestamp) we can use for regression
        df["date"] = pd.to_datetime(price_data["date"])
        df["date"] = df["date"].map(pd.Timestamp.timestamp)

        df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

        #Feature scaling
        scaler = MinMaxScaler()

        df_scaled = scaler.fit_transform(df[['date', 'price']])
        df_scaled = pd.DataFrame(df_scaled, columns =['date', 'price'])
    
        # Prepare data for LSTM
        n_steps = 30  #Number of time steps
        n_future_steps=10
        x, y = create_lstm_data(df_scaled, target_col='price', n_steps = n_steps, n_future_steps = n_future_steps)
        # Split the data into training set and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Reshape x for LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

        # Convert data to PyTorch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Reshape y_train and y_test to match the output shape (batch_size, 1)
        y_train = y_train.view(-1, 1)
        y_test = y_test.view(-1, 1)


        # Define Transformer model parameters
        input_dim = x_train.shape[2]  # Number of features (2 in this case: date and price)
        model_dim = 32  # Embedding size
        num_heads = 2  # Number of attention heads
        num_layers = 2  # Number of stacked transformer layers
        output_dim = 1  # Predicting one future price value
        dropout = 0.1

        # Initialize the Transformer model
        model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim, dropout)

        # Loss function and optimizer
        criterion = nn.MSELoss()  # Mean squared error for regression
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        epochs = 20
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            # Validate the model on the test set
            model.eval()
            with torch.no_grad():
                test_output = model(x_test)
                val_loss = criterion(test_output, y_test)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}")


        # Convert predictions and true values to NumPy arrays for evaluation
        y_test_np = y_test.numpy()
        test_output_np = test_output.numpy()

        # Compute Mean Squared Error (MSE)
        mse = mean_squared_error(y_test_np, test_output_np)
        print(f"Mean Squared Error (MSE): {mse}")

        # Compute R-squared (R²)
        r_squared = r2_score(y_test_np, test_output_np)
        print(f"R-squared (R²): {r_squared}")

        # Save the trained model
        torch.save(model.state_dict(), model_file_path)
        print(f"Trained model saved to {model_file_path}")

def get_inference(token, timeframe, region, data_provider):
    # Load the trained Transformer model
    model = TransformerModel(input_dim=2, model_dim=32, num_heads=2, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()


     # Load and preprocess the data
    price_data = pd.read_csv(training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to numerical timestamp and calculate price
    df["date"] = pd.to_datetime(price_data["date"])
    df["date"] = df["date"].map(pd.Timestamp.timestamp)
    df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    # Feature scaling
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['date', 'price']])

    # Prepare the latest sequence for prediction
    n_steps = 30
    if len(df_scaled) < n_steps:
        raise ValueError("Not enough data to make a prediction. Need at least {} rows, but got {}.".format(n_steps, len(df_scaled)))

    latest_data = df_scaled[-n_steps:]  # Get the latest n_steps data
    latest_sequence = np.expand_dims(latest_data, axis=0)  # Shape: (1, n_steps, 2)

    # Convert the latest sequence to a PyTorch tensor
    latest_sequence = torch.tensor(latest_sequence, dtype=torch.float32)

    # Predict the next price
    with torch.no_grad():
        current_price_pred = model(latest_sequence)

    # Reverse the scaling for the predicted price
    last_price_value = latest_data[-1, 1]  # Last price value (used for reverse scaling)

    # Combine last_date_value with predicted price
    pred_data = np.array([[last_price_value, current_price_pred[0, 0]]])  # Shape (1, 2)
    
     # Create a new scaler for reverse scaling, using 'price' column only
    scaler_price = MinMaxScaler()
    scaler_price.fit(df[['price']])  # Ensure this fits to the 'price' column used during training
    pred_data_unscaled = scaler_price.inverse_transform(pred_data)


    # Extract the actual predicted price
    predicted_price = pred_data_unscaled[0, 1]  # The unscaled predicted price


    return predicted_price

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(model_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        self.input_fc = nn.Linear(input_dim, model_dim)

    def forward(self, src):
        src = self.input_fc(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc1(output[:, -1, :])  # Last time step for each batch
        output = torch.relu(output)
        output = self.fc2(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

