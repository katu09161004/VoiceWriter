import tkinter as tk
from gui import VoiceWriterGUI

def main():
    root = tk.Tk()
    gui = VoiceWriterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()