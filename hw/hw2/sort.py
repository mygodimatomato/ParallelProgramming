def parse_line(line):
    """Parse the line and return a tuple with the extracted values."""
    parts = line.split(',')
    thread = parts[0].strip().split(' ')[-1]  # Extract thread identifier
    start = int(parts[1].strip().split(' ')[-1])  # Extract start value
    return thread, start, line

def sort_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Parse each line and sort
    sorted_lines = sorted([parse_line(line) for line in lines], key=lambda x: (x[0], x[1]))

    # Extract the original lines in sorted order
    sorted_lines = [line[2] for line in sorted_lines]
    
    return sorted_lines

# Replace 'yourfile.txt' with your filename
sorted_lines = sort_file('2b_test.txt')

# Output the sorted lines
for line in sorted_lines:
    print(line, end='')

# Optionally, write the sorted lines back to a file
with open('2b_test_sorted.txt', 'w') as file:
    file.writelines(sorted_lines)
