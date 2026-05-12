import platform

NUM_WORKERS = 0 if platform.system() == "Windows" else 4

def main():
    print("hello world")


if __name__ == "__main__":
    main()