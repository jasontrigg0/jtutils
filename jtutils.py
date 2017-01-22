#!/usr/bin/env python
import date
import itertools
import re

def to_days(dt_str):
    if dt_str == "": return ""
    return date.Date(dt_str).to_days()

def to_years(dt_str):
    if dt_str == "": return ""
    return date.Date(dt_str).to_years()

def is_int(var):
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
    if isinstance(s, str):
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
    return itertools.izip(a, b)

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
    for _xa, _xb, _xc in itertools.izip(prepend(None,a), b, postpend(None,c)):
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

    lesspager = None
    use_less = False
    if has_term and (wide or len(first_rows) == MAX_CAT_ROWS):
        use_less = True
        lesspager = LessPager()

    for l in itertools.chain(first_rows, lines):
        if use_less:
            lesspager.write(l + "\n")
        else:
            sys.stdout.write(l + "\n")


class LessPager(object):
    """
    Use for streaming writes to a less process
    Taken from pydoc.pipepager:
    /usr/lib/python2.7/pydoc.py
    """
    def __init__(self):
        self.proc = os.popen("less -S", 'w')
    def write(self, text):
        try:
            self.proc.write(text)
        except IOError:
            self.proc.close()
            sys.exit()
