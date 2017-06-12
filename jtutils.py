#!/usr/bin/env python
from __future__ import absolute_import
from . import date
import itertools
import re
import os
import sys
import io
import codecs
import requests
import bs4

def open_py2_py3(f):
    if f == sys.stdin:
        if sys.version_info[0] >= 3:
            f_in = codecs.getreader('utf8')(sys.stdin.detach(), errors='ignore')
        else:
            f_in = sys.stdin
    else:
        if sys.version_info[0] >= 3:
            f_in = open(f, errors='ignore')
        else:
            f_in = open(f)
    return f_in

def pd_read_csv(f, **args):
    #In python3 pd.read_csv is breaking on utf8 encoding errors
    #Solving this by reading the file into StringIO first and
    #then passing that into the pd.read_csv() method
    import pandas as pd
    import sys
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    f = StringIO(open_py2_py3(f).read())
    return pd.read_csv(f, **args)

def df_to_bytestrings(df):
    #avoid bug where pandas applymap() turns length 0 dataframe into a series
    if len(df) == 0:
        return df
    else:
        #convert the columns as well
        df.columns = [to_bytestring(c) for c in df.columns]
        return df.applymap(to_bytestring)

def to_bytestring(obj):
    """avoid encoding errors when writing!"""
    if isinstance(obj, str):
        return unicode(obj, errors="ignore").encode("ascii","ignore")
    elif isinstance(obj, unicode):
        return obj.encode("ascii","ignore")
    elif isinstance(obj, list):
        return str([to_bytestring(e) for e in obj])
    else:
        return obj

def to_days(dt_str):
    if dt_str == "": return ""
    return date.Date(dt_str).to_days()

def to_years(dt_str):
    if dt_str == "": return ""
    return date.Date(dt_str).to_years()

class GroupBy:
    def __init__(self, list_of_inputs, key, value=None):
        self.key = key
        if (not value):
            self.value = lambda x: x
        else:
            self.value = value
        self.dictionary = {}
        self.update(list_of_inputs)
    def update(self, l):
        for x in l:
            k = self.key(x)
            v = self.value(x)
            self.dictionary[k] = self[k] + [v]
        return self
    def __setitem__(self, key, value):
        raise Exception("Can't set counter items")
    def __getitem__(self, x):
        if x in self.dictionary:
            return self.dictionary[x]
        else:
            return []
    def __str__(self):
        return self.dictionary.__str__()
    def keys(self):
        return self.dictionary.keys()
    def values(self):
        return self.dictionary.values()
    def items(self):
        return self.dictionary.items()

def is_int(var):
    if sys.version_info[0] >= 3:
        return isinstance(var, int)
    else:
        return isinstance( var, ( int, long ) )

def str_is_int(var):
    # if not isinstance(var, str) and np.isnan(var):
    #     return False
    if re.findall("^\d+$",var):
        return True
    else:
        return False

def str_is_float(var):
    try:
        f = float(var)
        # if np.isnan(f):
        #     return False
        return True
    except:
        return False

def md5hash(s):
    import md5
    return md5.md5(s).hexdigest()

def rand():
    import random
    return str(round(random.random(),4))

def utf8_string(s):
    if sys.version_info[0] >= 3:
        #http://stackoverflow.com/questions/34869889/what-is-the-proper-way-to-determine-if-an-object-is-a-bytes-like-object-in-pytho
        if isinstance(s,str):
            return s
        else:
            return s.decode()
        return str(s, "utf-8")
    elif isinstance(s, str):
        return s.decode("utf-8","ignore").encode("utf-8","ignore")
    elif isinstance(s, unicode):
        return s.encode("utf-8","ignore")
    else:
        raise

def fix_broken_pipe():
    #following two lines solve 'Broken pipe' error when piping
    #script output into head
    from signal import signal, SIGPIPE, SIG_DFL
    signal(SIGPIPE,SIG_DFL)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    if (sys.version_info[0] >= 3):
        zip_fn = zip
    else:
        zip_fn = itertools.izip(a,b)
    return zip_fn(a,b)

def threewise(iterable):
    """s -> (None, s0, s1), (s0, s1, s2), ... (sn-1, sn, None)
    example:
    for (last, cur, next) in threewise(l):
    """
    a, b, c = itertools.tee(iterable,3)
    def prepend(val, l):
        yield val
        for i in l: yield i
    def postpend(val, l):
        for i in l: yield i
        yield val
    next(c,None)
    if (sys.version_info[0] >= 3):
        zip_fn = zip
    else:
        zip_fn = itertools.izip

    for _xa, _xb, _xc in zip_fn(prepend(None,a), b, postpend(None,c)):
        yield (_xa, _xb, _xc)

def terminal_size():
    try:
        columns = os.popen('tput cols').read().split()[0]
        return int(columns)
    except:
        return None

def lines2less(lines):
    """
    input: lines = list / iterator of strings
    eg: lines = ["This is the first line", "This is the second line"]

    output: print those lines to stdout if the output is short + narrow
            otherwise print the lines to less
    """
    lines = iter(lines) #cast list to iterator

    #print output to stdout if small, otherwise to less
    has_term = True
    terminal_cols = 100
    try:
        terminal_cols = terminal_size()
    except:
        #getting terminal info failed -- maybe it's a
        #weird situation like running through cron
        has_term = False

    MAX_CAT_ROWS = 20  #if there are <= this many rows then print to screen

    first_rows = list(itertools.islice(lines,0,MAX_CAT_ROWS))
    wide = any(len(l) > terminal_cols for l in first_rows)

    use_less = False
    if has_term and (wide or len(first_rows) == MAX_CAT_ROWS):
        use_less = True

    lines = itertools.chain(first_rows, lines)
    if sys.version_info[0] >= 3:
        map_fn = map
    else:
        map_fn = itertools.imap
    lines = map_fn(lambda x: x + '\n', lines)

    if use_less:
        lesspager(lines)
    else:
        for l in lines:
            sys.stdout.write(l)


def lesspager(lines):
    """
    Use for streaming writes to a less process
    Taken from pydoc.pipepager:
    /usr/lib/python2.7/pydoc.py
    and
    /usr/lib/python3.5/pydoc.py
    """
    cmd = "less -S"
    if sys.version_info[0] >= 3:
        """Page through text by feeding it to another program."""
        import subprocess
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
        try:
            with io.TextIOWrapper(proc.stdin, errors='backslashreplace') as pipe:
                try:
                    for l in lines:
                        pipe.write(l)
                except KeyboardInterrupt:
                    # We've hereby abandoned whatever text hasn't been written,
                    # but the pager is still in control of the terminal.
                    pass
        except OSError:
            pass # Ignore broken pipes caused by quitting the pager program.
        while True:
            try:
                proc.wait()
                break
            except KeyboardInterrupt:
                # Ignore ctl-c like the pager itself does.  Otherwise the pager is
                # left running and the terminal is in raw mode and unusable.
                pass

    else:
        proc = os.popen(cmd, 'w')
        try:
            for l in lines:
                proc.write(l)
        except IOError:
            proc.close()
            sys.exit()

def argmax(l,f=None):
    """http://stackoverflow.com/questions/5098580/implementing-argmax-in-python"""
    if f:
        l = [f(i) for i in l]
    return max(enumerate(l), key=lambda x:x[1])[0]

#website functions
def html_to_soup(html):
    if isinstance(html, str):
        html = html.decode("utf-8","ignore")
    try:
        soup = bs4.BeautifulSoup(html, "lxml")
    except:
        soup = bs4.BeautifulSoup(html, "html.parser")
    return soup

def url_to_soup(url, js=False, encoding=None):
    html = _get_webpage(url, js, encoding)
    return html_to_soup(html)

def _get_webpage(url, js=False, encoding = None):
    if js:
        return _get_webpage_with_js(url)
    else:
        return _get_webpage_static(url, encoding)

def _get_webpage_with_js(url):
    with open_driver() as driver:
        driver.get(url)
        wait_until_stable(driver)
        return driver.page_source

def _get_webpage_static(url, encoding=None):
    if not url.startswith("http"):
        url = "http://" + url
    headers = {'User-agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:40.0) Gecko/20100101 Firefox/40.0'}
    s = requests.Session()
    RETRIES = 5
    for i in range(RETRIES):
        try:
            out = s.get(url, headers=headers, timeout=(10,10))
            if encoding:
                out.encoding = encoding
            return out.text
        except (requests.exceptions.RequestException, requests.Timeout, requests.exceptions.ReadTimeout) as e:
            if i < (RETRIES - 1):
                continue
            else:
                raise e


def run(cmd):
    import subprocess
    pipes = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = pipes.communicate()
    return_code = pipes.returncode
    return stdout, stderr, return_code
