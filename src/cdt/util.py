import datetime

def str2dur(string):
    """
    Note:
        \d+s: \d seconds
        \d+m: \d minutes
        \d+h: \d hours
        \d+d: \d days
        \d+w: \d * 7 days
    """
    if "s" in string:
        num = int(string.partition("s")[0])
        return datetime.timedelta(seconds = num)
    elif "m" in string:
        num = int(string.partition("m")[0])
        return datetime.timedelta(minutes = num)
    elif "h" in string:
        num = int(string.partition("h")[0])
        return datetime.timedelta(hours = num)
    elif "d" in string:
        num = int(string.partition("d")[0])
        return datetime.timedelta(days = num)
    elif "w" in string:
        num = int(string.partition("w")[0])
        return datetime.timedelta(days = num * 7)
    else:
        raise ValueError("Duration string invalid")