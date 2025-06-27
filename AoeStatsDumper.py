import requests
import os
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class AoeStatsDumper:
    _BASE_URL = "https://aoestats.io"
    _DUMP_URL = f"{_BASE_URL}/api/db_dumps"

    def __init__(self, data_path="data", raw_data_path="raw_data", verbose=False):
        self.data_path = Path(data_path)
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("AoeStatsDumper")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)

    def fetch_dumps_index(self):
        self.logger.info(f"[GET] Fetching dump index from {self._DUMP_URL} ...")
        resp = requests.get(self._DUMP_URL)
        resp.raise_for_status()
        return resp.json()["db_dumps"]

    def _is_valid_parquet(self, path):
        try:
            pd.read_parquet(path, engine="auto", columns=[])
            return True
        except Exception as e:
            self.logger.warning(f"[BROKEN] {path} is not readable: {e}")
            return False

    def _download_file(self, url, save_path):
        """
        Загружает бинарный файл по URL и сохраняет его по указанному пути.
        """
        try:
            self.logger.info(f"[GET] {url}")
            resp = requests.get(url)
            resp.raise_for_status()
            save_path.write_bytes(resp.content)
            self.logger.info(f"[OK] Saved to {save_path}")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to download {url}: {e}")
            if save_path.exists():
                save_path.unlunk()

    def download_all_raw_data(self, force=False, max_workers=4):
        dumps = self.fetch_dumps_index()
        tasks = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for dump in dumps:
                date_range = f'{dump["start_date"]}_{dump["end_date"]}'
                for key in ['players_url', 'matches_url']:
                    url = self._BASE_URL + dump[key]
                    key_name = key.replace("_url", "")
                    filename = f"{date_range}_{key_name}.parquet"
                    save_path = self.raw_data_path / filename

                    need_download = force or not save_path.exists() or not self._is_valid_parquet(save_path)
                    if not need_download:
                        self.logger.info(f"[SKIP] {save_path} already exists.")
                        continue

                    tasks.append(executor.submit(self._download_file, url, save_path))
            for _ in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading parquet files"):
                pass


if __name__ == "__main__":
    dumper = AoeStatsDumper()
    dumper.download_all_raw_data()
