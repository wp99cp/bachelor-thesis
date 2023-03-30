import os
import sys
from multiprocessing import Lock

# import the necessary packages form the pre-processing/image_splitter
sys.path.insert(0, os.environ['BASE_DIR'] + '/helper-scripts/python_helpers')
# noinspection PyUnresolvedReferences
from singleton import singleton


@singleton
class SentinelMemoryManager:

    def __init__(self, lock=Lock()):
        print("Memory Manager initialized! This should only happen once.")

        # this is a global variable, so that the dates are only opened once
        self.__open_dates = {}
        self.__memory_lock = lock

    def close_date(self, date: str):
        """
        Closes the date used to save memory.

        :param date: the date
        """

        assert date in self.__open_dates, "Date not open, please open it first."

        with self.__memory_lock:
            # check if counter is 1, if so, close the date
            if self.__open_dates[date]['counter'] == 1:
                del self.__open_dates[date]
            # otherwise, just decrement the counter
            else:
                self.__open_dates[date]['counter'] -= 1

    def has_date(self, date: str):
        """
        Checks if the date is open.

        :param date: the date
        :return: True if the date is open, False otherwise
        """

        with self.__memory_lock:
            return date in self.__open_dates.keys()

    def get_date_data(self, date: str):
        """
        Returns the data for the given date.

        :param date: the date
        :return: the data
        """

        assert date in self.__open_dates, "Date not open, please open it first."

        with self.__memory_lock:
            self.__open_dates[date]['counter'] += 1

            return self.__open_dates[date]

    def add_date_data(self, date: str, data: dict):
        """
        Sets the date for which the patches should be created.

        :param data:
        :param date: the date
        """

        assert date in self.__open_dates, "Date not open, please open it first."

        # synchronize the access to the open_dates dictionary
        with self.__memory_lock:
            counter = self.__open_dates[date]['counter']
            self.__open_dates[date] = data
            self.__open_dates[date]['counter'] = counter + 1

    def add_date(self, date: str):
        """
        Sets the date to loaded.

        :param date: the date
        """

        assert date not in self.__open_dates.keys(), "Date already open!"

        with self.__memory_lock:
            self.__open_dates[date] = {
                'counter': 1,
            }

    def get_open_dates(self):
        """
        Returns the open dates.

        :return: the open dates
        """

        return self.__open_dates.keys()
