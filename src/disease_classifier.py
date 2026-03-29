from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np


def train_model(features, labels):
    """
    Train a simple classifier for disease detection.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print("Model Accuracy:", accuracy)

    return model


if __name__ == "__main__":
    features = np.random.rand(100, 15)
    labels = np.random.randint(0, 2, 100)

    model = train_model(features, labels)

if __name__ == "__main__":
    from voice_feature_extraction import extract_features

    audio_path = "data/sample_voice.wav"

    features = extract_features(audio_path)

    import numpy as np

    X = np.array([features, features])
    y = np.array([0, 1])

    train_model(X, y)
