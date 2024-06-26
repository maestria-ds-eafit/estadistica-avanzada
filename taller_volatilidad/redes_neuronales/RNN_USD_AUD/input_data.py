import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class InputData:
    def __init__(
        self,
        data_path: str,
        sequence_length: int,
        train_size_proportion=0.8,
        scaler: MinMaxScaler | StandardScaler | None = None,
    ):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.train_size_proportion = train_size_proportion
        self.scaler = scaler
        self.read_data()
        self.scale_data()

    def read_data(self):
        df = pd.read_csv(self.data_path)
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df.set_index("Fecha", inplace=True)
        self.df = df[["USD_AUD"]]
        self.data = df["USD_AUD"].to_numpy().reshape(-1, 1)

    def scale_data(self):
        if not self.scaler:
            return

        self.data = self.scaler.fit_transform(self.df)

    def create_training_and_test_sets(self):
        X, y = [], []
        for i in range(len(self.data) - self.sequence_length):
            X.append(self.data[i : i + self.sequence_length])
            y.append(self.data[i + self.sequence_length])
        X, y = np.array(X), np.array(y)
        self.train_size = int(self.train_size_proportion * len(X))
        X_train, X_test = X[: self.train_size], X[self.train_size :]
        y_train, y_test = y[: self.train_size], y[self.train_size :]
        return X_train, X_test, y_train, y_test
