from datetime import datetime

def get_duration_hours(start_time):
    """
    Prints and returns the time difference in hours

    :param start_time: datetime object
    :return: time difference in hours
    """
    time_diff = datetime.now() - start_time
    time_diff_hours = time_diff.seconds / 3600
    print('hours:', round(time_diff_hours, 2))
    return time_diff_hours