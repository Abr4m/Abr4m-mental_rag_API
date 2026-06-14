"""
Repositories: CSVSuggestionRepository
Loads and pre-processes the mental health CSV dataset.
Implements the SuggestionRepository port.
Previously at infrastructure/data/csv_repository.py.
"""
from pathlib import Path
from typing import List

import pandas as pd

from app.models.interfaces import SuggestionRepository


class CSVSuggestionRepository(SuggestionRepository):
    def __init__(self, csv_path: str | Path) -> None:
        self._path = Path(csv_path)
        self._df_sug: pd.DataFrame | None = None
        self._concern_meta: pd.DataFrame | None = None
        self._tags: List[str] | None = None

    # ── Lazy loading ──────────────────────────────────────────────────── #
    def _load(self) -> None:
        if self._df_sug is not None:
            return  # already loaded

        df = pd.read_csv(self._path)
        df = df[["Concern", "Suggestion", "tag", "multi_labels", "title"]].dropna()

        # Suggestion-level df (one row = one suggestion)
        self._df_sug = df.rename(columns={"Suggestion": "text"}).reset_index(drop=True)

        # Concern-level metadata (for source cards)
        all_labels = (
            df.groupby("Concern")["multi_labels"]
            .agg(lambda x: ",".join(sorted(set(",".join(x).split(",")))))
            .reset_index()
            .rename(columns={"multi_labels": "all_labels"})
        )
        meta = (
            df.groupby("Concern", sort=False)
            .agg(tag=("tag", "first"), title=("title", "first"))
            .reset_index()
        )
        self._concern_meta = meta.merge(all_labels, on="Concern").reset_index(drop=True)
        self._tags = ["All"] + sorted(df["tag"].dropna().unique().tolist())

    # ── SuggestionRepository interface ────────────────────────────────── #
    def get_suggestion_dataframe(self) -> pd.DataFrame:
        self._load()
        return self._df_sug  # type: ignore[return-value]

    def get_concern_meta(self) -> pd.DataFrame:
        self._load()
        return self._concern_meta  # type: ignore[return-value]

    def get_available_tags(self) -> List[str]:
        self._load()
        return self._tags  # type: ignore[return-value]
