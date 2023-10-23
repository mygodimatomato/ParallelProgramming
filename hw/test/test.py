from math import sqrt, ceil

# for radius in range(0, 100):
radius = 3
r = radius
x = 0
y = r
d = 1-r
count = 0
while x <= y:
  count += y
  if d < 0:
    d += 2 * x + 3
    x += 1
  else:
    d += 2 * (x - y) + 5
    x += 1
    y -= 1
  
pixels = 0
for i in range (0, radius):
  print(i, ceil(sqrt(radius * radius - i * i)))
  pixels += ceil(sqrt(radius * radius - i * i))
print("radius = ", radius, "test = ", count, "traditional = ", pixels)