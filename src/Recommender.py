"""
Spotify Recommendation Engine
-----------------------------

Author: Cynthia Mueni
Project: Spotify SRecommendation System
"""

import numpy as np
import pandas as pd


class ContentBasedRecommender:
    """
    Spotify Recommendation Engine.

    Supports:
    1. Popularity-Based Recommendations (Baseline)
    2. Content-Based Recommendations (Cosine Similarity)
    """

    def __init__(self, tracks_df, similarity_matrix, track_lookup):
        """
        Initialize the recommender.

        Parameters
        ----------
        tracks_df : pandas.DataFrame
            Dataset containing processed track information.

        similarity_matrix : numpy.ndarray
            Cosine similarity matrix.

        track_lookup : pandas.Series
            Maps track_key to DataFrame index.
        """

        self.tracks_df = tracks_df
        self.similarity_matrix = similarity_matrix
        self.track_lookup = track_lookup

    # ============================================================
    # Available Songs
    # ============================================================

    def get_available_tracks(self):
        """
        Return all available track keys.

        Returns
        -------
        list
        """

        return sorted(self.tracks_df["track_key"].tolist())

    # ============================================================
    # Popularity-Based Recommender (Baseline)
    # ============================================================

    def popularity_recommendations(self, top_n=10, candidate_df=None):
        """
        Recommend the most preferred songs.

        Parameters
        ----------
        top_n : int

        candidate_df : pandas.DataFrame, optional
            A pre-filtered subset of tracks_df to rank instead of the
            full dataset (used to apply sidebar filters).

        Returns
        -------
        pandas.DataFrame
        """

        base = candidate_df if candidate_df is not None else self.tracks_df

        recommendations = (
            base
            .sort_values(
                by="preference_score",
                ascending=False
            )
            .head(top_n)
            .copy()
        )

        recommendations.insert(
            0,
            "Rank",
            range(1, len(recommendations) + 1)
        )

        return recommendations[
            [
                "Rank",
                "track_key",
                "track_name",
                "artist_name",
                "preference_score",
                "play_count",
                "total_minutes_played",
                "playlist_count",
                "in_library"
            ]
        ]

    # ============================================================
    # Content-Based Recommender
    # ============================================================

    def content_recommendations(self, track_key, top_n=10, candidate_indices=None):
        """
        Recommend songs similar to one or more selected tracks.

        Parameters
        ----------
        track_key : str or list of str
            A single seed track, or several seed tracks. When several are
            given, their similarity scores are averaged so the
            recommendations reflect all of the seeds together.

        top_n : int

        candidate_indices : iterable of int, optional
            Restrict recommendations to these row positions in
            `tracks_df` (used to apply sidebar filters before ranking).
            If None, every track is eligible.

        Returns
        -------
        pandas.DataFrame
        """

        seed_keys = [track_key] if isinstance(track_key, str) else list(track_key)

        missing = [k for k in seed_keys if k not in self.track_lookup.index]
        if missing:
            raise ValueError(f"Track(s) not found: {missing}")

        seed_indices = [self.track_lookup[k] for k in seed_keys]

        # Average similarity across all seed songs
        combined_scores = np.mean(
            [self.similarity_matrix[idx] for idx in seed_indices],
            axis=0
        )

        eligible = (
            set(candidate_indices) if candidate_indices is not None
            else set(range(len(self.tracks_df)))
        )
        eligible -= set(seed_indices)

        similarity_scores = sorted(
            ((i, combined_scores[i]) for i in eligible),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        recommended_indices = [
            i[0] for i in similarity_scores
        ]

        recommendations = (
            self.tracks_df
            .iloc[recommended_indices]
            .copy()
        )

        recommendations["similarity_score"] = [
            score for _, score in similarity_scores
        ]

        recommendations.insert(
            0,
            "Rank",
            range(1, len(recommendations) + 1)
        )

        recommendations["recommendation_reason"] = recommendations.apply(
            self._generate_reason,
            axis=1
        )

        return recommendations[
            [
                "Rank",
                "track_key",
                "track_name",
                "artist_name",
                "similarity_score",
                "recommendation_reason",
                "preference_score",
                "play_count",
                "total_minutes_played",
                "playlist_count",
                "in_library"
            ]
        ]

    # ============================================================
    # Recommendation Reason
    # ============================================================

    def _generate_reason(self, row):
        """
        Generate a simple explanation for the recommendation.
        """

        if row["preference_score"] >= 0.80:
            return "Top‑tier listening choice"

        elif row["playlist_count"] >= 2:
            return "Frequently curated into collections"

        elif row["in_library"] == 1:
            return "Added to your personal library"

        else:
            return "Similar to tracks you enjoy"

    # ============================================================
    # Unified Recommendation Interface
    # ============================================================

    def recommend(self,
                  method="content",
                  track_key=None,
                  top_n=10,
                  candidate_df=None,
                  candidate_indices=None):
        """
        Generate recommendations.

        Parameters
        ----------
        method : str

            "content"
            "popularity"

        track_key : str or list of str
            One or more seed tracks (content-based only).

        top_n : int

        candidate_df : pandas.DataFrame, optional
            Pre-filtered subset to rank (popularity method).

        candidate_indices : iterable of int, optional
            Pre-filtered row positions to restrict recommendations to
            (content method).
        """

        method = method.lower()

        if method == "popularity":
            return self.popularity_recommendations(top_n, candidate_df=candidate_df)

        elif method == "content":

            if track_key is None:
                raise ValueError(
                    "track_key must be provided for content recommendations."
                )

            return self.content_recommendations(
                track_key,
                top_n,
                candidate_indices=candidate_indices
            )

        else:
            raise ValueError(
                "Method must be either 'content' or 'popularity'."
            )