import random
import datetime

# Global Lists
name = []
phno = []
add = []
checkin = []
checkout = []
room = []
price = []
rc = []
p = []
roomno = []
custid = []
day = []

# Global Index
i = 0

# Home Function
def Home():
    print("\n" + "-" * 60)
    print("\t\tWELCOME TO HOTEL ANCASA")
    print("-" * 60)
    print("1. Booking")
    print("2. Rooms Info")
    print("3. Room Service (Menu Card)")
    print("4. Payment")
    print("5. Record")
    print("0. Exit")
    print("-" * 60)

    ch = input("Enter your choice: ")

    if ch == '1':
        Booking()
    elif ch == '2':
        Rooms_Info()
    elif ch == '3':
        restaurant()
    elif ch == '4':
        Payment()
    elif ch == '5':
        Record()
    elif ch == '0':
        print("Thank you for visiting Hotel AnCasa!")
        exit()
    else:
        print("Invalid choice. Try again.")
        Home()


# Booking Function
def Booking():
    global i

    print("\n--- ROOM BOOKING ---")

    while True:
        n = input("Name: ").strip()
        p1 = input("Phone No.: ").strip()
        a = input("Address: ").strip()

        if n and p1 and a:
            name.append(n)
            phno.append(p1)
            add.append(a)
            break
        else:
            print("Name, Phone No. and Address cannot be empty!")

    try:
        cii = input("Check-In (dd/mm/yyyy): ")
        cii_split = list(map(int, cii.split('/')))
        checkin.append(cii)

        coo = input("Check-Out (dd/mm/yyyy): ")
        coo_split = list(map(int, coo.split('/')))
        checkout.append(coo)

        d1 = datetime.datetime(cii_split[2], cii_split[1], cii_split[0])
        d2 = datetime.datetime(coo_split[2], coo_split[1], coo_split[0])
        delta = (d2 - d1).days

        if delta <= 0:
            raise ValueError("Check-out date must be after check-in.")

        day.append(delta)
    except Exception as e:
        print("Invalid date format or logic:", e)
        name.pop()
        phno.pop()
        add.pop()
        checkin.pop()
        checkout.pop()
        Booking()
        return

    print("\n--- SELECT ROOM TYPE ---")
    print("1. Standard Non-AC - Rs. 3500")
    print("2. Standard AC - Rs. 4000")
    print("3. 3-Bed Non-AC - Rs. 4500")
    print("4. 3-Bed AC - Rs. 5000")

    try:
        ch = int(input("Enter choice: "))
        room_types = ["Standard Non-AC", "Standard AC", "3-Bed Non-AC", "3-Bed AC"]
        prices = [3500, 4000, 4500, 5000]

        if 1 <= ch <= 4:
            room.append(room_types[ch - 1])
            price.append(prices[ch - 1])
        else:
            raise ValueError("Invalid room type.")
    except:
        print("Invalid input. Restarting booking...")
        name.pop()
        phno.pop()
        add.pop()
        checkin.pop()
        checkout.pop()
        Booking()
        return

    rn = random.randint(300, 340)
    cid = random.randint(10, 50)

    while rn in roomno or cid in custid:
        rn = random.randint(300, 340)
        cid = random.randint(10, 50)

    roomno.append(rn)
    custid.append(cid)
    rc.append(0)
    p.append(0)

    print("\n*** ROOM BOOKED SUCCESSFULLY ***")
    print("Room No:", rn)
    print("Customer ID:", cid)

    i += 1
    input("Press Enter to return to Home...")
    Home()


# Rooms Info Function
def Rooms_Info():
    print("\n--- ROOMS INFORMATION ---")
    print("1. Standard Non-AC: 1 Double Bed, TV, Balcony, Rs. 3500")
    print("2. Standard AC: Same as Non-AC + AC, Rs. 4000")
    print("3. 3-Bed Non-AC: 1 Double + 1 Single Bed, Rs. 4500")
    print("4. 3-Bed AC: Same as Non-AC + AC, Rs. 5000")
    input("\nPress Enter to return to Home...")
    Home()


# Placeholder for restaurant
def restaurant():
    print("\n--- RESTAURANT MENU ---")
    print("This section is under development.")
    input("\nPress Enter to return to Home...")
    Home()


# Placeholder for Payment
def Payment():
    print("\n--- PAYMENT ---")
    print("This section is under development.")
    input("\nPress Enter to return to Home...")
    Home()


# Placeholder for Record
def Record():
    print("\n--- CUSTOMER RECORD ---")
    if not name:
        print("No bookings yet.")
    else:
        for idx in range(i):
            print(f"\nCustomer ID: {custid[idx]}, Name: {name[idx]}, Room: {room[idx]}, Days: {day[idx]}")
    input("\nPress Enter to return to Home...")
    Home()


# Start the App
if __name__ == "__main__":
    Home()
