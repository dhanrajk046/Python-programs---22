tasks = []

def show_tasks():
    if not tasks:
        print("No tasks in the list.")
    else:
        print("To-Do List:")
        for idx, task in enumerate(tasks, 1):
            print(f"{idx}. {task}")

def add_task():
    task = input("Enter a new task: ")
    tasks.append(task)
    print("Task added.")

def update_task():
    show_tasks()
    idx = int(input("Enter the task number to update: ")) - 1
    if 0 <= idx < len(tasks):
        new_task = input("Enter the updated task: ")
        tasks[idx] = new_task
        print("Task updated.")
    else:
        print("Invalid task number.")

def delete_task():
    show_tasks()
    idx = int(input("Enter the task number to delete: ")) - 1
    if 0 <= idx < len(tasks):
        tasks.pop(idx)
        print("Task deleted.")
    else:
        print("Invalid task number.")

while True:
    print("\n1. Show Tasks\n2. Add Task\n3. Update Task\n4. Delete Task\n5. Exit")
    choice = input("Choose an option: ")

    if choice == '1':
        show_tasks()
    elif choice == '2':
        add_task()
    elif choice == '3':
        update_task()
    elif choice == '4':
        delete_task()
    elif choice == '5':
        print("Goodbye!")
        break
    else:
        print("Invalid choice. Please try again.")