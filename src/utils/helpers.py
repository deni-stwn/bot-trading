def calculate_moving_average(data, period):
    if len(data) < period:
        return None
    return sum(data[-period:]) / period

def calculate_rsi(data, period=14):
    if len(data) < period:
        return None
    gains = []
    losses = []
    
    for i in range(1, period + 1):
        change = data[-i] - data[-(i + 1)]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            losses.append(-change)
            gains.append(0)

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def format_price(price):
    return f"{price:.2f}"

def is_trade_signal(rsi, overbought=70, oversold=30):
    if rsi is None:
        return None
    if rsi > overbought:
        return "sell"
    elif rsi < oversold:
        return "buy"
    return "hold"