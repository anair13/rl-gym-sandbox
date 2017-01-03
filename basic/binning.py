def safe_bin(value, min, max, bins):
    assert value < max, "%f %f %f %f" % (value, min, max, bins)
    assert value >= min, "%f %f %f %f" % (value, min, max, bins)
    bin_size = float(max - min) / bins
    return int((value - min) / bin_size)

def unsafe_bin(value, min, max, bins):
    if value <= min:
    	return 0
    if value >= max:
    	return bins - 1
    bin_size = float(max - min) / bins
    return int((value - min) / bin_size)