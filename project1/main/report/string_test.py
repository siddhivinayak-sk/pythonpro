

def test_string():
    value = convert_ascii_numbers_to_string([100, 101, 112])
    print(f"Value: {value}")


def convert_ascii_numbers_to_string(chars):
    return "".join([chr(c) for c in chars])


if __name__ == "__main__":
    test_string()
