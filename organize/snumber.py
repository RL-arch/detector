def series_number(filename):
    for i in range(100):
        if f"s{str(i)}t" in filename:
            return str(i)
        elif f"s0{str(i)}t" in filename:
            return str(i)
