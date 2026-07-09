"""
Spotify Recommendation Engine
-----------------------------

Author: Cynthia Mueni
Project: Spotify SRecommendation System
"""

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

    def popularity_recommendations(self, top_n=10):
        """
        Recommend the most preferred songs.

        Parameters
        ----------
        top_n : int

        Returns
        -------
        pandas.DataFrame
        """

        recommendations = (
            self.tracks_df
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
                "Rank"
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

    def content_recommendations(self, track_key, top_n=10):
        """
        Recommend songs similar to a selected track.

        Parameters
        ----------
        track_key : str

        top_n : int

        Returns
        -------
        pandas.DataFrame
        """

        if track_key not in self.track_lookup.index:
            raise ValueError(f"'{track_key}' not found.")

        idx = self.track_lookup[track_key]

        similarity_scores = list(
            enumerate(
                self.similarity_matrix[idx]
            )
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        similarity_scores = similarity_scores[1:top_n + 1]

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
                "Rank"
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
                  top_n=10):
        """
        Generate recommendations.

        Parameters
        ----------
        method : str

            "content"
            "popularity"

        track_key : str

        top_n : int
        """

        method = method.lower()

        if method == "popularity":
            return self.popularity_recommendations(top_n)

        elif method == "content":

            if track_key is None:
                raise ValueError(
                    "track_key must be provided for content recommendations."
                )

            return self.content_recommendations(
                track_key,
                top_n
            )

        else:
            raise ValueError(
                "Method must be either 'content' or 'popularity'."
            )