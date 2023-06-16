import os
import sys
from multiprocessing import Lock

# import the necessary packages form the pre-processing/image_splitter
sys.path.insert(0, os.environ['BASE_DIR'] + '/helper-scripts/python_helpers')
# noinspection PyUnresolvedReferences
from singleton import singleton


@singleton
class SentinelMemoryManager:

    def __init__(self, lock=Lock(), max_concurrent_open_dates=4):
        print("Memory Manager initialized! This should only happen once.")

        # this is a global variable, so that the dates are only opened once
        self.__open_dates = {}
        self.__memory_lock = lock

        # the maximum number of dates that can be opened at the same time
        self.__least_recently_used = []
        self.__max_concurrent_open_dates = max_concurrent_open_dates

    def close_date(self, date: str):
        """
        Closes the date used to save memory.

        :param date: the date
        """

        with self.__memory_lock:
            del self.__least_recently_used[self.__least_recently_used.index(date)]
            self.__close_date(date)

    def __close_date(self, date: str):
        """
        Closes the date used to save memory.
        This method is not thread safe, so it should only be called from within a lock.

        :param date: the date
        """

        assert date in self.__open_dates, "Date not open, please open it first."

        # check if counter is 1, if so, close the date
        del self.__open_dates[date]

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
            # update the least recently used list
            self.__least_recently_used.remove(date)
            self.__least_recently_used.append(date)

            return self.__open_dates[date]

    def add_date_data(self, date: str, data: dict):
        """
        Sets the date for which the patches should be created.

        :param data:
        :param date: the date
        """

        assert date in self.__open_dates.keys(), "Date not open, please open it first."

        # synchronize the access to the open_dates dictionary
        with self.__memory_lock:
            # update the least recently used list
            self.__least_recently_used.remove(date)
            self.__least_recently_used.append(date)

            self.__open_dates[date] = data

    def add_date(self, date: str):
        """
        Sets the date to loaded.

        :param date: the date
        """

        assert date not in self.__open_dates.keys(), "Date already open!"

        self.__memory_lock.acquire()

        # update the least recently used list
        self.__least_recently_used.append(date)

        # check if the maximum number of dates is reached
        if len(self.__least_recently_used) > self.__max_concurrent_open_dates:
            print("«« [Memory Manager]: Closing date " + self.__least_recently_used[0])

            # close the least recently used date
            self.__close_date(self.__least_recently_used[0])
            del self.__least_recently_used[0]

        # open the date
        self.__open_dates[date] = {'loading': True}

        print("»» [Memory Manager]: Opening date " + date)
        self.__memory_lock.release()

    def get_open_dates(self):
        """
        Returns the open dates.

        :return: the open dates
        """

        return self.__open_dates.keys()
