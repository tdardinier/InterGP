def quantize(x, n_div, mini, maxi):
    y = (x - mini) / (maxi - mini)
    y = min(0.99, max(0, y))
    return int(y * n_div)
