import os

"""
File Handling
File open mode:
"r" - Read - Default value. Opens a file for reading, error if the file does not exist
"a" - Append - Opens a file for appending, creates the file if it does not exist
"w" - Write - Opens a file for writing, creates the file if it does not exist
"x" - Create - Creates the specified file, returns an error if the file exists

Additional handling mode:
"t" - Text - Default value. Text mode
"b" - Binary - Binary mode (e.g. images)
"""


def file_handing():
    text_source = "this is a text to create a text file"
    file_name = "text_file.txt"

    # Check if file exists
    if os.path.exists(file_name):
        print("File already exists, going to delete it")
        os.remove(file_name)

    # Create a file
    with open(file_name, "w") as file:
        file.write(text_source)

    # Read a file
    with open(file_name, "r") as file:
        text_read = file.read()
        print(text_read)
