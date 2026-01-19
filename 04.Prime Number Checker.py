def is_prime(num):
    if num <= 1:
        return False  # 0 and 1 are not prime numbers
    for i in range(2, int(num ** 0.5) + 1):  # Check divisibility up to square root of num
        if num % i == 0:
            return False  # If divisible, not a prime
    return True  # If not divisible by any number, it's a prime

number = int(input("Enter a number: "))

if is_prime(number):
    print(f"{number} is a prime number.")
else:
    print(f"{number} is not a prime number.")