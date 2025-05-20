import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid


def calculateStatistics(data, isSample=True):
    """
    Compute descriptive statistics and goodness-of-fit test for a numeric dataset.

    Args:
        data (list): Numeric list or array.
        isSample (bool): Use sample statistics if True; population otherwise.

    Returns:
        dict: Dictionary of statistical metrics.
    """
    try:
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
        ddof = 1 if isSample else 0
        variance = np.var(data, ddof=ddof)
        stdDeviation = np.std(data, ddof=ddof)
        coefVariation = stdDeviation / mean if mean != 0 else None
        skewness = stats.skew(data, bias=not isSample)
        kurtosis = stats.kurtosis(data, bias=not isSample)
        quartiles = {f"q{i}": np.percentile(data, i * 25) for i in range(5)}
        iqr = quartiles["q3"] - quartiles["q1"]
        percentiles = {f"p{p}": np.percentile(data, p) for p in range(0, 101, 5)}
        ksResults = goodnessOfFitKSTest(data)

        return makeSerializable(
            {
                "count": n,
                "mean": roundSig(mean),
                "sum": roundSig(totalSum),
                "median": roundSig(median),
                "mode": [roundSig(m) for m in mode],
                "modeCount": [roundSig(c) for c in modeCount],
                "min": roundSig(dataMin),
                "max": roundSig(dataMax),
                "range": roundSig(dataRange),
                "variance": roundSig(variance),
                "standardDeviation": roundSig(stdDeviation),
                "coefficientOfVariation": roundSig(coefVariation),
                "skewness": roundSig(skewness),
                "kurtosis": roundSig(kurtosis),
                "quartiles": {k: roundSig(v) for k, v in quartiles.items()},
                "iqr": roundSig(iqr),
                "percentiles": {k: roundSig(v) for k, v in percentiles.items()},
                "kolmogorovTest": ksResults,
            }
        )

    except Exception as e:
        print("Error in calculateStatistics:", e)
        fallbackData = np.random.normal(0, 1, 50).tolist()
        return calculateStatistics(fallbackData, isSample=True)


def generateRandomData(n, distribution):
    """
    Generate synthetic numeric data for testing.

    Args:
        n (int): Number of elements.
        distribution (str): Distribution type to sample from.

    Returns:
        list: Random data.
    """
    try:
        n = int(n)
        if distribution == "normal":
            return np.random.normal(loc=0, scale=1, size=n).tolist()
        elif distribution == "uniform":
            return np.random.uniform(low=0, high=1, size=n).tolist()
        elif distribution == "exponential":
            return np.random.exponential(scale=1.0, size=n).tolist()
        elif distribution == "binomial":
            return np.random.binomial(n=10, p=0.5, size=n).tolist()
        elif distribution == "poisson":
            return np.random.poisson(lam=3, size=n).tolist()
        else:
            raise ValueError("Unsupported distribution")
    except Exception as e:
        print("Error in generateRandomData:", e)
        return np.random.normal(0, 1, 50).tolist()


def extractFirstNumericColumn(filePath):
    """
    Extract the first numeric column from a CSV file.

    Args:
        filePath (str): Path to the CSV file.

    Returns:
        list: Cleaned numeric data.
    """
    try:
        df = pd.read_csv(filePath)
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                return df[column].dropna().tolist()
        raise ValueError("No numeric column found.")
    except Exception as e:
        print("Error in extractFirstNumericColumn:", e)
        return np.random.normal(0, 1, 50).tolist()


def generatePlots(data, outputDir="static/plots"):
    """
    Generate histogram and boxplot for the given data.

    Args:
        data (list): Numeric data.
        outputDir (str): Directory to save plots.

    Returns:
        tuple: Paths to saved histogram and boxplot images.
    """
    try:
        os.makedirs(outputDir, exist_ok=True)
        plotId = str(uuid.uuid4())

        histPath = os.path.join(outputDir, f"hist_{plotId}.png")
        plt.figure()
        sns.histplot(data, kde=True, bins=20)
        plt.title("Histogram")
        plt.savefig(histPath)
        plt.close()

        boxPath = os.path.join(outputDir, f"box_{plotId}.png")
        plt.figure()
        sns.boxplot(x=data)
        plt.title("Boxplot")
        plt.savefig(boxPath)
        plt.close()

        return histPath, boxPath

    except Exception as e:
        print("Error in generatePlots:", e)
        return None, None


def makeSerializable(obj):
    """
    Convert numpy and complex objects to native Python types for serialization.

    Args:
        obj: Any object.

    Returns:
        Native Python equivalent.
    """
    if isinstance(obj, dict):
        return {k: makeSerializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [makeSerializable(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def roundSig(x, sig=2):
    """
    Round a number to a specified number of significant digits.

    Args:
        x (float or int): Number to round.
        sig (int): Significant digits.

    Returns:
        float: Rounded number.
    """
    if isinstance(x, (int, float, np.number)):
        return float(f"{x:.{sig}g}")
    return x


def goodnessOfFitKSTest(data):
    """
    Apply Kolmogorov-Smirnov test for multiple distributions and sort by p-value.

    Args:
        data (list): Numeric data.

    Returns:
        dict: Distribution names mapped to their p-values (rounded).
    """
    commonDistributions = [
        "norm",
        "uniform",
        "expon",
        "lognorm",
        "gamma",
        "beta",
        "laplace",
        "t",
        "pareto",
        "cauchy",
        "triang",
    ]

    results = {}

    for distName in commonDistributions:
        try:
            dist = getattr(stats, distName)
            params = dist.fit(data)
            stat, pValue = stats.kstest(data, distName, args=params)
            results[distName] = roundSig(pValue)
        except Exception:
            continue

    sortedResults = dict(
        sorted(results.items(), key=lambda item: item[1], reverse=True)
    )
    return sortedResults
