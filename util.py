import datetime
import ujson


def log(msg):
    """Logs message to std out w/ time stamp."""

    def fmt(t):
        return '{:02d}'.format(t)

    now = datetime.datetime.now()
    print('{}-{}-{} {}:{}:{} {}'.format(fmt(now.year), fmt(now.month), fmt(now.day), fmt(now.hour),
                                        fmt(now.minute), fmt(now.second), msg), flush=True)


def load_lines(doc_path_in):
    """Loads json lines.
    :param doc_path_in: The document from which to load the lines.
    :returns dictionaries: JSON lines as dictionaries.
    """
    lines = open(doc_path_in, 'r').readlines()
    dictionaries = [ujson.loads(line) for line in lines]
    return dictionaries
