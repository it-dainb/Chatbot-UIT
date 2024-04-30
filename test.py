test = "0.5a"

print(all([char.isnumeric() for char in test.split(".")]))

print(test.isnumeric())