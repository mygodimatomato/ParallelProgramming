def draw_circle(xc, yc, x, y):
    # This function represents drawing pixels on the screen.
    # In a real scenario, you would replace print statements with drawing pixels on the screen.
    print(xc + x, yc + y)
    print(xc - x, yc + y)
    print(xc + x, yc - y)
    print(xc - x, yc - y)
    print(xc + y, yc + x)
    print(xc - y, yc + x)
    print(xc + y, yc - x)
    print(xc - y, yc - x)

def bresenham_circle(xc, yc, r):
    x = 0
    y = r
    d = 1 - r  # Initial decision parameter
    print(r)
    while y > x:
      if d < 0:
        d += 2 * x + 3
        x += 1
      else:
        d += 2 * (x - y) + 5
        x += 1
        y -= 1
      print(y)

# Example usage:
xc, yc = 0, 0  # Center of the circle
r = 3  # Radius of the circle
bresenham_circle(xc, yc, r)