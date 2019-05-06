import time

print("DolARK loading:  ", end="\r")

symbs = ["|", "\\", "-", "/"]
i = 0

while True:
    time.sleep(0.5)
    i = (i+1) % 4
    print(f"DolARK loading: {symbs[i]}", end="\r")
