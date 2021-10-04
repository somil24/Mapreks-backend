import os

def next_path(path_pattern):
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b

def create_dirs(path):
    try:
        path = next_path(path)
        os.mkdir(path)
        input_path = path + '/input'
        output_path = path + '/output'
        os.mkdir(input_path)
        os.mkdir(output_path)
        return input_path, output_path
    except Exception as e:
        print('\n\n\nerror creating dirs: ', e)