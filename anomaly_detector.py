import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras import regularizers
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    print("[WARNING] TensorFlow failed to load:", e)
    TENSORFLOW_AVAILABLE = False

class AnomalyDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = pd.DataFrame(index=df.index)
        self.models_run = []
        self.numeric_data = self.df.select_dtypes(include=[np.number]).dropna()
        self.scaler = StandardScaler()

    def detect_statistical(self):
        z_scores = np.abs(zscore(self.numeric_data, nan_policy='omit'))
        z_flags = (z_scores > 3).any(axis=1).astype(int)
        self.results.loc[self.numeric_data.index, 'z_score_flag'] = z_flags
        self.models_run.append("z_score")

    def detect_isolation_forest(self):
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(self.numeric_data)
        self.results.loc[self.numeric_data.index, 'isolation_forest_flag'] = (preds == -1).astype(int)
        self.models_run.append("isolation_forest")

    def detect_lof(self):
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        preds = lof.fit_predict(self.numeric_data)
        self.results.loc[self.numeric_data.index, 'lof_flag'] = (preds == -1).astype(int)
        self.models_run.append("lof")

    def detect_one_class_svm(self):
        svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
        preds = svm.fit_predict(self.numeric_data)
        self.results.loc[self.numeric_data.index, 'svm_flag'] = (preds == -1).astype(int)
        self.models_run.append("svm")

    def detect_autoencoder(self):
        if not TENSORFLOW_AVAILABLE:
            return
            raise RuntimeError("TensorFlow is not available in this environment.")
        # Normalize the data
        data_scaled = self.scaler.fit_transform(self.numeric_data)

        # Define autoencoder architecture
        input_dim = data_scaled.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(16, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_layer)
        bottleneck = Dense(8, activation="relu")(encoded)
        decoded = Dense(16, activation="relu")(bottleneck)
        output_layer = Dense(input_dim, activation="linear")(decoded)

        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer="adam", loss="mse")

        # Train the autoencoder
        autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)

        # Get reconstruction error
        reconstructed = autoencoder.predict(data_scaled)
        mse = np.mean(np.power(data_scaled - reconstructed, 2), axis=1)

        # Threshold: top 5% highest MSE = anomalies
        threshold = np.percentile(mse, 95)
        auto_flags = (mse > threshold).astype(int)

        self.results.loc[self.numeric_data.index, 'autoencoder_flag'] = auto_flags
        self.models_run.append("autoencoder")

    def combine_results(self):
        flag_cols = [col for col in self.results.columns if col.endswith('_flag')]
        self.results['anomaly_score'] = self.results[flag_cols].sum(axis=1)
        self.results['consensus_flag'] = (self.results['anomaly_score'] >= 2).astype(int)
        return self.results

    def run_auto(self):
        num_cols = self.numeric_data.shape[1]
        if num_cols == 1:
            self.detect_statistical()
        elif 1 < num_cols < 15:
            self.detect_statistical()
            self.detect_isolation_forest()
            self.detect_lof()
            self.detect_autoencoder()
        else:
            self.detect_statistical()
            self.detect_isolation_forest()
            self.detect_lof()
            self.detect_one_class_svm()
            self.detect_autoencoder()
        return self.combine_results()