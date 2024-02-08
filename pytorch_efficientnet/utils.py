"""

"""
from config import *


# utilities
class AverageMeter(object):
    """
    computes and stores the average and current value of metrics over time
    :param val (float): The current value.
    :param avg (float): The average of all updates.
    :param sum (float): The sum of all values updated.
    :param count (int): The number of updates.
    """

    def __init__(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        """reset all values to 0"""
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val, n=1):
        """
        update the metrics with a new value.

        :param val (float): The value to add.
        :param n (int): The weight of update, the batch size

        Returns:
            float: The updated average.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def as_minutes(s: float):
    """
    convert to minutes
    :param s:
    :return:
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since: float, percent: float):
    """
    time delta
    :param since:
    :param percent:
    :return: time delta
    """
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))  # return time delta


def get_logger(filename=PATHS.OUTPUT_DIR):
    """
    logging function
    :param filename:
    :return:
    """
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}v{CFG.VERSION}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed: int):
    """
    seed everything
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sep():
    """prints a separator line"""
    print("-" * 100)