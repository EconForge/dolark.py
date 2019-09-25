import time

print("DolARK loading:  ", end="\r")

symbs = ["|", "\\", "-", "/"]
i = 0

I = 10
for i in range(I):
    time.sleep(0.5)
    i = (i+1) % 4
    print(f"DolARK loading: {symbs[i]}", end="\r")
