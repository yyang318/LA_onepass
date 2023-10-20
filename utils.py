import math


def list_sum(list_):
    sum_ = 0
    for ele in list_:
        sum_ += ele
    return sum_

def list_mean(list_):
    sum_ = 0
    for ele in list_:
        sum_ += ele
    return sum_ * 1.0 / len(list_)


def list_std(list_):
    mean = list_mean(list_)
    second_moment = 0
    for ele in list_:
        second_moment += (ele - mean) ** 2
    second_moment /= (1.0 * len(list_))
    return math.sqrt(second_moment)

def list_std_2d(list_, axis=None):
    if axis is None:
        mean_ = list_mean_2d(list_)
        count = 0.0
        second_moment = 0.0
        for row in list_:
            for ele in row:
                second_moment += (ele - mean_) ** 2
                count += 1
        second_moment /= count
        return math.sqrt(second_moment)
    elif axis == 0:
        mean_ = list_mean_2d(list_,0)
        std = []
        num_rows = len(list_)
        num_cols = len(list_[0])
        for col in range(num_cols):
            second_moment = 0.0
            count = 0.0
            for row in range(num_rows):
                second_moment += (list_[row][col] - mean_[col]) ** 2
                count += 1
            second_moment /= count
            std.append(math.sqrt(second_moment))
        return std
    elif axis == 1:
        mean_ = list_mean_2d(list_,1)
        std = []
        num_rows = len(list_)
        num_cols = len(list_[0])
        for row in range(num_rows):
            second_moment = 0.0
            count = 0.0
            for col in range(num_cols):
                second_moment += (list_[row][col] - mean_[row]) ** 2
                count += 1
            second_moment /= count
            std.append(math.sqrt(second_moment))
        return std
    return None


def list_mean_2d(list_, axis= None):
    if axis is None:
        sum_ = 0
        count = 0
        for row in list_:
            for ele in row:
                sum_ += ele
                count += 1.0
        return sum_/count
    if axis == 0:
        #sum over columns
        sum_ = []
        num_rows = len(list_)
        num_cols = len(list_[0])
        for col in range(num_cols):
            sum_col = 0
            for row in range(num_rows):
                sum_col += list_[row][col]
            sum_.append(sum_col/num_rows)
        return sum_
    elif axis == 1:
        #sum over rows: sum the elements of each row
        sum_ = []
        num_rows = len(list_)
        num_cols = len(list_[0])
        for row in range(num_rows):
            sum_row = 0
            for col in range(num_cols):
                sum_row += list_[row][col]
            sum_.append(sum_row/num_cols)
        return sum_

