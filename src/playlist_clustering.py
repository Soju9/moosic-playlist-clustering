from typing import Iterable, Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


DEFAULT_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def load_tracks(path: str) -> pd.DataFrame:
    tracks = pd.read_csv(path)
    tracks.columns = tracks.columns.str.strip()
    return tracks


def prepare_features(
    tracks: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    selected_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    features = tracks[selected_columns].copy()
    return features.apply(pd.to_numeric, errors="coerce").dropna()


def scale_features(features: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    return pd.DataFrame(scaled, columns=features.columns, index=features.index)


def fit_kmeans(features: pd.DataFrame, n_clusters: int, random_state: int = 200) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(features)
    return model


def evaluate_cluster_range(
    features: pd.DataFrame,
    min_k: int = 2,
    max_k: int = 30,
    random_state: int = 200,
) -> pd.DataFrame:
    rows = []
    for k in range(min_k, max_k + 1):
        model = fit_kmeans(features, n_clusters=k, random_state=random_state)
        rows.append(
            {
                "n_clusters": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(features, model.labels_),
            }
        )
    return pd.DataFrame(rows)


def attach_playlist_labels(tracks: pd.DataFrame, labels) -> pd.DataFrame:
    playlist_df = tracks.copy().reset_index(drop=True)
    playlist_df["cluster"] = labels
    return playlist_df


def summarize_playlists(playlist_df: pd.DataFrame) -> pd.DataFrame:
    return (
        playlist_df.groupby("cluster")
        .agg(
            song_count=("name", "count"),
            sample_songs=("name", lambda values: ", ".join(values.head(5))),
            sample_artists=("artist", lambda values: ", ".join(values.head(5))),
        )
        .reset_index()
        .sort_values("cluster")
    )

