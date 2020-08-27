def write_to_file(file_path, s):
    with open(file_path, "a") as f:
        f.write(s)