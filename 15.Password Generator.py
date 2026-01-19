import string
import random

# Get password length
while True:
    try:
        length = int(input("Enter password length: "))
        if length < 1:
            print("Please enter a positive number.")
            continue
        break
    except ValueError:
        print("Please enter a valid integer.")

print('''Choose character sets for your password. 
You can select several options (for each desired set):
    1. Letters (uppercase & lowercase)
    2. Digits
    3. Special characters
    4. Done selecting character sets
''')

characterList = ""

while True:
    try:
        choice = int(input("Pick a number (1-4): "))
        if choice == 1:
            characterList += string.ascii_letters
            print("Letters added.")
        elif choice == 2:
            characterList += string.digits
            print("Digits added.")
        elif choice == 3:
            characterList += string.punctuation
            print("Special characters added.")
        elif choice == 4:
            if characterList == "":
                print("You must add at least one character set!")
                continue
            break
        else:
            print("Please pick a valid option (1-4)!")
    except ValueError:
        print("Please enter a number (1-4).")

# Optionally remove duplicate characters (in case user picks the same option multiple times)
characterList = ''.join(sorted(set(characterList), key=characterList.index))

# Generate password
password = ''.join(random.choice(characterList) for _ in range(length))

print("The random password is:", password)
