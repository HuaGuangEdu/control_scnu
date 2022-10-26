import math
from numbers import Number
import random

def sin(angular:float):
    """
    求出输入角度angular的sin值
    Args:
        angular: 输入角度

    Returns: 输出sin值

    """
    return math.sin(angular / 180.0 * math.pi)

def cos(angular:float):
    """
    求出输入角度angular的cos值
    Args:
        angular: 输入角度

    Returns: 输出cos值

    """
    return math.cos(angular / 180.0 * math.pi)

def tan(angular:float):
    """
    求出输入角度angular的tan值
    Args:
        angular: 输入角度

    Returns: 输出tan值

    """
    return math.tan(angular / 180.0 * math.pi)

def asin(value:float):
    """
    求出输入值的asin值
    Args:
        angular: 输入值

    Returns: 输出asin值

    """
    return math.asin(value) / math.pi * 180

def acos(value:float):
    """
    求出输入值的acos值
    Args:
        angular: 输入值

    Returns: 输出acos值

    """
    return math.acos(value) / math.pi * 180

def atan(value:float):
    """
    求出输入值的atan值
    Args:
        angular: 输入值

    Returns: 输出atan值

    """
    return math.atan(value) / math.pi * 180

def math_isPrime(n:Number):
    """
    判断n是否为质数
    Args:
        angular: 输入数值

    Returns: 返回True或False

    """
    if not isinstance(n, Number):
        try:
            n = float(n)
        except:
            return False
    if n == 2 or n == 3:
        return True
    # False if n is negative, is 1, or not whole, or if n is divisible by 2 or 3.
    if n <= 1 or n % 1 != 0 or n % 2 == 0 or n % 3 == 0:
        return False
    # Check all the numbers of form 6k +/- 1, up to sqrt(n).
    for x in range(6, int(math.sqrt(n)) + 2, 6):
        if n % (x - 1) == 0 or n % (x + 1) == 0:
            return False
    return True

def Sum(List:list):
    """
    求出输入列表内各个元素之和
    Args:
        angular: 输入列表

    Returns: 输出结果

    """
    if not isinstance(List,list):
        raise ValueError("输入的必须是列表")
    return sum(List)

def Min(List:list):
    """
    求出输入列表内最小的元素
    Args:
        angular: 输入列表

    Returns: 输出结果

    """
    if not isinstance(List,list):
        raise ValueError("输入的必须是列表")
    return min(List)

def Max(List:list):
    """
    求出输入列表内最大的元素
    Args:
        angular: 输入列表

    Returns: 输出结果

    """
    if not isinstance(List,list):
        raise ValueError("输入的必须是列表")
    return max(List)

def mean(myList:list):
    """
    求出输入列表内各个元素的平均值
    Args:
        angular: 输入列表

    Returns: 输出结果

    """
    if not isinstance(myList, list):
        raise ValueError("输入的必须是列表")
    localList = [e for e in myList if isinstance(e, Number)]
    if not localList: return
    return float(sum(localList)) / len(localList)

def median(myList:list):
    """
    求出输入列表内各个元素中的中位数
    Args:
        angular: 输入列表

    Returns: 输出结果

    """
    if not isinstance(myList, list):
        raise ValueError("输入的必须是列表")
    localList = sorted([e for e in myList if isinstance(e, Number)])
    if not localList: return
    if len(localList) % 2 == 0:
        return (localList[len(localList) // 2 - 1] + localList[len(localList) // 2]) / 2.0
    else:
        return localList[(len(localList) - 1) // 2]

def modes(some_list:list):
    """
    求出输入列表内各个元素中的众数
    Args:
        angular: 输入列表

    Returns: 输出结果

    """
    if not isinstance(some_list, list):
        raise ValueError("输入的必须是列表")
    modes = []
    counts = []
    maxCount = 1
    for item in some_list:
        found = False
        for count in counts:
            if count[0] == item:
                count[1] += 1
                maxCount = max(maxCount, count[1])
                found = True
        if not found:
            counts.append([item, 1])
    for counted_item, item_count in counts:
        if item_count == maxCount:
            modes.append(counted_item)
    return modes

def standard_deviation(numbers:list):
    """
    求出输入列表内各个元素的的标准差
    Args:
        angular: 输入列表

    Returns: 输出结果

    """
    if not isinstance(numbers, list):
        raise ValueError("输入的必须是列表")
    n = len(numbers)
    if n == 0: return
    mean = float(sum(numbers)) / n
    variance = sum((x - mean) ** 2 for x in numbers) / n
    return math.sqrt(variance)

def choice(List:list):
    """
    从列表中随机抽取一个值出来
    Args:
        angular: 输入列表

    Returns: 随机抽取的值

    """
    if not isinstance(List, list):
        raise ValueError("输入的必须是列表")
    return random.choice(List)
