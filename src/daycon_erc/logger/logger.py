"""
logger 설정

Description:
    logger 설정을 위한 시간 포멧, 시간 기준 정의 및 설정

Author:
    Name: Kyunghyun Lim
    Email: fly1294@naver.com
"""

import datetime
import logging
import os

import pytz


class Formatter(logging.Formatter):
    """
    Description:
        override logging.Formatter to use an aware datetime object
    """

    @staticmethod
    def converter(timestamp: str) -> datetime:
        """
        Description:
            로깅 시간 한국 시간으로 설정
        """
        # Create datetime in UTC
        dt = datetime.datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        # Change datetime's timezone
        return dt.astimezone(pytz.timezone("Asia/Seoul"))

    def formatTime(self, record, datefmt=None) -> str:
        """
        Description:
            시간 출력 형식 포멧팅
        """
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="milliseconds")
            except TypeError:
                s = dt.isoformat()
        return s


def set_logger() -> logging:
    """
    Description:
        로깅 객체 및 포맷 정의
    """

    path = "./output/logs/"
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + str(datetime.datetime.now()) + ".txt"
    logging.basicConfig(filename=file_name, level=logging.INFO)

    mylogger = logging.getLogger("process")
    mylogger.setLevel(logging.INFO)
    mylogger.propagate = False

    formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(filename=file_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    mylogger.addHandler(fh)
    mylogger.addHandler(stream_handler)

    return mylogger
