from tkinter import *
import tkinter.messagebox

def entertask():
    def add():
        input_text = entry_task.get(1.0, "end-1c").strip()
        if not input_text:
            tkinter.messagebox.showwarning("Warning!", "Please enter some text")
        else:
            listbox_task.insert(END, input_text)
            add_window.destroy()

    add_window = Toplevel(window)
    add_window.title("Add Task")
    entry_task = Text(add_window, width=40, height=4)
    entry_task.pack(padx=10, pady=10)
    button_temp = Button(add_window, text="Add task", command=add)
    button_temp.pack(pady=5)

def deletetask():
    selected = listbox_task.curselection()
    if not selected:
        tkinter.messagebox.showwarning("Warning", "Please select a task to delete")
        return
    listbox_task.delete(selected[0])

def markcompleted():
    marked = listbox_task.curselection()
    if not marked:
        tkinter.messagebox.showwarning("Warning", "Please select a task to mark as completed")
        return
    current_text = listbox_task.get(marked[0])
    # Only add mark if not already marked
    if not current_text.endswith(" ✔"):
        listbox_task.delete(marked[0])
        listbox_task.insert(marked[0], current_text + " ✔")

# Main window
window = Tk()
window.title("To-Do List APPLICATION")

# Frame for listbox and scrollbar
frame_task = Frame(window)
frame_task.pack()

listbox_task = Listbox(
    frame_task, bg="black", fg="white", height=15,
    width=50, font="Helvetica"
)
listbox_task.pack(side=LEFT, fill=BOTH)

scrollbar_task = Scrollbar(frame_task)
scrollbar_task.pack(side=RIGHT, fill=Y)
listbox_task.config(yscrollcommand=scrollbar_task.set)
scrollbar_task.config(command=listbox_task.yview)

entry_button = Button(window, text="Add task", width=50, command=entertask)
entry_button.pack(pady=3)

delete_button = Button(window, text="Delete selected task", width=50, command=deletetask)
delete_button.pack(pady=3)

mark_button = Button(window, text="Mark as completed", width=50, command=markcompleted)
mark_button.pack(pady=3)

window.mainloop()
