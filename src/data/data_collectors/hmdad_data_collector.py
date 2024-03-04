import time

import requests

from src.config import HMDAD_RAW_DATASET_DIRECTOR
from src.utils import prj_logger
from src.utils.io import json_dump

collector_session = requests.Session()

logger = prj_logger.getLogger(__name__)


class HMDADDataCollectorManager:
    DATA_API_URL = "https://www.cuilab.cn/files/dmi/data_download.txt"
    TABLE_NAME = 'microbe_disease'

    def update_tables(self):
        self._update_microbe_disease_table(self.TABLE_NAME)
        logger.info(f"table with name {self.TABLE_NAME} saved!")

    def _update_microbe_disease_table(self, table_name):
        data = self.fetch_data(self.DATA_API_URL)
        json_dump(f"{HMDAD_RAW_DATASET_DIRECTOR}/{table_name}.json", data)

    @staticmethod
    def fetch_data(url):
        result = None
        while url:
            time.sleep(0.05)
            try:
                response = collector_session.get(url=url, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    data_content = response.content
                    items = [[item for item in row.split('\\t')] for row in str(data_content)[2:-3].split('\\n')]
                    header = items[0]
                    result = [{header[j]: item[j] for j in range(len(header))} for item in items[1:]]
                    logger.info(f"request with url {url} completed.")
                    url = None
                else:
                    logger.warning(f"request with url {url} failed.")
            except Exception as e:
                print(f"exception in response of {url} : {e}")

        return result
