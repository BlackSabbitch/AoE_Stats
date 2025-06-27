from pathlib import Path
from typing import Tuple, List
from datetime import date, datetime
import pandas as pd
import logging


class AoeDataLoader:
    def __init__(self, raw_data_path="raw_data", verbose=True):
        self.raw_data_path = Path(raw_data_path)
        self.matches_files = self._get_parquet_files('matches')
        self.players_files = self._get_parquet_files('players')
        self.date_ranges = self.list_available_dates(base='matches')
        assert self.date_ranges == self.list_available_dates(base='players')

        self.logger = logging.getLogger("AoeDataLoader")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)

    def _get_parquet_files(self, kind="matches"):
        return sorted(self.raw_data_path.glob(f"*_{kind}.parquet"))

    def list_available_dates(self, base='matches') -> List[Tuple[date, date]]:
        dates = []
        files = self.matches_files if base == 'matches' else self.players_files
        for file in files:
            start_str, end_str = file.name.split(f"_{base}")[0].split('_')
            dates.append(
                (
                    datetime.strptime(start_str, "%Y-%m-%d").date(),
                    datetime.strptime(end_str, "%Y-%m-%d").date()
                    )
                )
        return sorted(dates)

    def load_data(
        self,
        players: bool = True,
        matches: bool = True,
        force_reload: bool = False,
        merge: bool = False,
        matches_pre_clean: bool = True,
        players_pre_clean: bool = True,
        keep_invalid_game_id: bool  = True,
    ) -> dict:
        if not players and not matches:
            raise ValueError("At least one of 'players' or 'matches' must be True.")

        if not hasattr(self, "_cached_data") or force_reload:
            self._cached_data = {}

        result = {}

        for flag, key, files_list, need_pre_clean in zip(
            [matches, players],
            ["matches", "players"],
            [self.matches_files, self.players_files],
            [matches_pre_clean, players_pre_clean],):

            if flag:
                if key not in self._cached_data or force_reload:
                    df = pd.concat([pd.read_parquet(f) for f in files_list], ignore_index=True)
                    if need_pre_clean:
                        df = self._clean(df, keep_invalid_game_id)
                    self._cached_data[key] = df
                result[key] = self._cached_data[key]

        if merge:
            if "merged" not in self._cached_data or force_reload:
                if "players" not in result or "matches" not in result:
                    raise ValueError("Both players and matches data must be loaded to perform merge.")
                self._cached_data["merged"] = result["players"].merge(result["matches"], on="game_id", how="left", suffixes=("_player", "_match"))
            result["merged"] = self._cached_data["merged"]
            self.logger.info(f"[MERGE] Players and matches merged: {result['merged'].shape[0]} rows")

        return result

    def _clean(self, df: pd.DataFrame, keep_invalid_game_id: bool = True) -> pd.DataFrame:
        initial_shape = df.shape
        df = df.dropna(how='all').drop_duplicates()
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        if 'started_timestamp' in df.columns: # его нет в players, есть только в mathes
            df['started_timestamp'] = pd.to_datetime(df['started_timestamp'], errors='coerce')

        if 'num_players' in df.columns: # его нет в players, есть только в mathes
            df = df[df['num_players'].apply(lambda x: isinstance(x, int) and x > 1 and x % 2 == 0)]

        if not keep_invalid_game_id:
            df = df.dropna(subset=["game_id"])
            df = df[df["game_id"].apply(lambda x: isinstance(x, (str, int)))]
            df["game_id"] = df["game_id"].astype(str)

        final_shape = df.shape
        self.logger.info(f"[CLEAN] From {initial_shape} → {final_shape} after cleaning.")

        return df.reset_index(drop=True)

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.describe(include="all")


if __name__ == "__main__":
    loader = AoeDataLoader()
    print(loader.list_available_dates())
