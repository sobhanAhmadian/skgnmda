import requests
from src.utils import prj_logger
from src.utils.io import json_dump
from src.config import VMH_RAW_DATASET_DIRECTOR
import time
import numpy as np

collector_session = requests.Session()

logger = prj_logger.getLogger(__name__)


class VMHDataCollectorManager:
    TABLES_LIST_API_URL = "https://www.vmh.life/_api/"

    def __init__(self):
        self.tables_data = list()

    def collect_all_data(self):
        self.update_tables(self.get_tables_list())

    def get_tables_list(self):
        response = collector_session.get(url=self.TABLES_LIST_API_URL)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"request with url {self.TABLES_LIST_API_URL} failed.")

    def update_tables(self, tables_data):
        for table_name, api_url in tables_data.items():
            self._update_table(table_name, api_url)
            logger.info(f"table with name {table_name} saved!")

    def _update_table(self, table_name, api_url):
        data, _, _ = self.fetch_all_pages(api_url)
        json_dump(f"{VMH_RAW_DATASET_DIRECTOR}/{table_name}.json", data)

    @staticmethod
    def fetch_all_pages(url, start_page=0, end_page=np.Inf):
        result = list()
        next_page_url = url
        p = start_page
        while next_page_url and p <= end_page:
            time.sleep(0.05)
            try:
                response = collector_session.get(url=next_page_url, timeout=10)
                if response.status_code == 200:
                    result_dict = response.json()
                    result.extend(result_dict["results"])
                    logger.info(f"request with url {next_page_url} completed.")
                    next_page_url = result_dict["next"]
                    p += 1
                else:
                    logger.warning(f"request with url {next_page_url} failed.")
            except Exception as e:
                print(f"exception in response of {next_page_url} : {e}")

        return result, next_page_url, p
