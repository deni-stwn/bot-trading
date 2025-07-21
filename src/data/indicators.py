from typing import List

def moving_average(data: List[float], window: int) -> List[float]:
    """Calculate the moving average of a given data set."""
    if window <= 0:
        raise ValueError("Window size must be a positive integer.")
    if len(data) < window:
        raise ValueError("Data length must be greater than or equal to window size.")
    
    averages = []
    for i in range(len(data) - window + 1):
        window_data = data[i:i + window]
        averages.append(sum(window_data) / window)
    return averages

def rsi(data: List[float], period: int) -> List[float]:
    """Calculate the Relative Strength Index (RSI) of a given data set."""
    if period <= 0:
        raise ValueError("Period must be a positive integer.")
    if len(data) < period:
        raise ValueError("Data length must be greater than or equal to period size.")
    
    gains = []
    losses = []
    
    for i in range(1, len(data)):
        change = data[i] - data[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return [100] * len(data)  # RSI is 100 if no losses
    
    rs = avg_gain / avg_loss
    rsi_value = 100 - (100 / (1 + rs))
    
    rsi_values = [rsi_value] * (period - 1)  # Fill initial values with None or similar
    rsi_values.extend([100 - (100 / (1 + (sum(gains[i-period+1:i+1]) / sum(losses[i-period+1:i+1])))) for i in range(period - 1, len(data))])
    
    return rsi_values