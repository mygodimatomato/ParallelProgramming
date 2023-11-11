def parse_line(line):
    """Parse the line to extract the 'a' value."""
    parts = line.split(',')
    a_value = float(parts[0].split('=')[1].strip())
    return a_value, line

def sort_file(filename):
    """Sort the lines of a file based on the 'a' value."""
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Parse and sort lines
    sorted_lines = sorted([parse_line(line) for line in lines], key=lambda x: x[0])

    # Extract sorted lines without the 'a' value
    sorted_lines = [line[1] for line in sorted_lines]

    return sorted_lines

# Process the files
file1 = './2b_test_2.txt'

sorted_lines_file1 = sort_file(file1)

# Optionally write sorted lines back to the files or new files
with open('sorted_2b_test_2.txt', 'w') as f:
    f.writelines(sorted_lines_file1)