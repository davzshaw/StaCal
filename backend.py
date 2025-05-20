import numpy as np
from scipy import stats

def calculateStatistics(data, isSample=True):
    """
    Calculate basic and advanced statistics for a numeric dataset.
    
    Args:
        data (list or np.array): The dataset.
        isSample (bool): If True, calculates sample statistics. Otherwise, population stats.
    
    Returns:
        dict: Dictionary with all calculated statistics.
    """

    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    totalSum = np.sum(data)
    median = np.median(data)
    modeResult = stats.mode(data, keepdims=True)
    mode = modeResult.mode.tolist()
    modeCount = modeResult.count.tolist()
    dataMin = np.min(data)
    dataMax = np.max(data)
    dataRange = dataMax - dataMin
    sortedData = np.sort(data)

    # Variance and standard deviation
    ddof = 1 if isSample else 0
    variance = np.var(data, ddof=ddof)
    stdDeviation = np.std(data, ddof=ddof)

    # Coefficient of variation
    coefVariation = stdDeviation / mean if mean != 0 else None

    # Skewness and kurtosis
    skewness = stats.skew(data, bias=not isSample)
    kurtosis = stats.kurtosis(data, bias=not isSample)

    # Quartiles (Q0 to Q4)
    quartiles = {
        f"q{i}": np.percentile(data, i * 25) for i in range(5)
    }

    # IQR
    iqr = quartiles["q3"] - quartiles["q1"]

    # Percentiles (0% to 100% in steps of 5)
    percentiles = {
        f"p{p}": np.percentile(data, p) for p in range(0, 101, 5)
    }

    return {
        "count": n,
        "sum": totalSum,
        "mean": mean,
        "median": median,
        "mode": mode,
        "modeCount": modeCount,
        "min": dataMin,
        "max": dataMax,
        "range": dataRange,
        "sortedData": sortedData.tolist(),
        "variance": variance,
        "standardDeviation": stdDeviation,
        "coefficientOfVariation": coefVariation,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "quartiles": quartiles,
        "iqr": iqr,
        "percentiles": percentiles
    }
