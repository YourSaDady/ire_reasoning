import time

def convert_time(start_t):
    spent = time.time() - start_t
    hrs = int(spent // 3600)
    spent %= 3600
    mins = int(spent // 60)
    secs = int(spent % 60)
    return [hrs, mins, secs]