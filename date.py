#!/usr/bin/env python
import datetime

month_dict = {"JAN":"01",
              "FEB":"02",
              "MAR":"03",
              "APR":"04",
              "MAY":"05",
              "JUN":"06",
              "JUL":"07",
              "AUG":"08",
              "SEP":"09",
              "OCT":"10",
              "NOV":"11",
              "DEC":"12",
              "JANUARY":"01",
              "FEBRUARY":"02",
              "MARCH":"03",
              "APRIL":"04",
              "MAY":"05",
              "JUNE":"06",
              "JULY":"07",
              "AUGUST":"08",
              "SEPTEMBER":"09",
              "OCTOBER":"10",
              "NOVEMBER":"11",
              "DECEMBER":"12"
}

class Date(object):
    """
    Date object contains a datetime object and manipulates it
    """
    def __init__(self, dt_str):
        dt_str = str(dt_str)
        self.dt_obj = self.date_string_to_datetime(dt_str)
    @classmethod
    def parse_month(cls, month_str):
        try:
            return month_dict[month_str.upper()]
        except KeyError:
            raise Exception("Invalid month string: " + month_str)
    @classmethod
    def date_diff(cls, dt_str1, dt_str2):
        return cls(dt_str1).to_years() - cls(dt_str2).to_years()
    def to_YYYYMMDD(self):
        return self.dt_obj.strftime('%Y%m%d')
    def to_years(self):
        tt = self.dt_obj.timetuple()
        #use (tm_yday - 1)
        #or else 20131231 -> year=2013, day=365 --> return 2014.0
        return tt.tm_year + (tt.tm_yday - 1) / 365.
    def to_days(self):
        import dateutil.parser
        epoch = datetime.datetime.utcfromtimestamp(0)
        return (self.dt_obj - epoch).days
    @classmethod
    def date_string_to_datetime(cls, dt_str):
        import re
        #23JAN2014
        f1 = re.findall("^(\d{2})(\w{3})(\d{4})$",dt_str)
        if f1:
            f = f1[0]
            DD = int(f[0])
            MM = int(cls.parse_month(f[1]))
            YYYY = int(f[2])
            return datetime.datetime(YYYY, MM, DD)

        #07MAR2014:14:09:53
        f2 = re.findall("^(\d{2})(\w{3})(\d{4}):(\d{2}):(\d{2}):(\d{2})$",dt_str)
        if f2:
            f = f2[0]
            DD = int(f[0])
            MM = int(cls.parse_month(f[1]))
            YYYY = int(f[2])
            return datetime.datetime(YYYY, MM, DD)

        #20140227
        if re.findall("^\d{8}$",dt_str) and dt_str[:2] in ["19","20"]:
            YYYY = int(dt_str[:4])
            MM = int(dt_str[4:6])
            DD = int(dt_str[6:8])
            return datetime.datetime(YYYY, MM, DD)

        #2014-05-13
        if len(dt_str.split("-")) == 3:
            YYYY, MM, DD = dt_str.split("-")
            YYYY = int(YYYY)
            MM = int(MM)
            DD = int(DD)
            if 1 <= MM <= 12 and 1 <= DD <= 31:
                return datetime.datetime(YYYY, MM, DD)

        #22 September 1988
        if re.findall("\d+ \w+ \d{4}",dt_str):
            day, month, year = dt_str.split()
            YYYY = int(year)
            MM = int(cls.parse_month(month))
            DD = int(day)
            return datetime.datetime(YYYY, MM, DD)

        #Nov 1, 2016
        if re.findall("\w{3} \d+, \d{4}",dt_str):
            month, day = dt_str.split(",")[0].split()
            year = dt_str.split(",")[1].strip()
            YYYY = int(year)
            MM = int(cls.parse_month(month))
            DD = int(day)
            return datetime.datetime(YYYY, MM, DD)

        raise Exception("unknown date string format: '{dt_str}'".format(**vars()))
