x0 = -10
x1 = 10
y0 = -10
y1 = 10
w = 20
h = 20
a = 2
b = 1

unit_width = (x1 - x0) / w
unit_height = (y1 - y0) / h

# initialize the cursor position
X_cur = x0
Y_cur = y0
X_prev = X_cur
Y_prev = Y_cur

def det_blocks(x0, x1, y0, y1, w, h, a, b, X_cur, Y_cur, X_prev, Y_prev) :
    # check if the block is out of bound
    Y_prev = Y_cur
    X_prev = X_cur
    X_cur = X_cur + unit_width * a
    if X_cur >= x1 :
      X_cur = x0 + X_cur - x1
      Y_cur = Y_cur + unit_height * b
      if Y_cur >= y1 :
        X_prev = X_cur
        Y_prev = Y_cur
        X_cur = 100
        Y_cur = 100
    
  return X_cur, Y_cur, X_prev, Y_prev

# run a for loop to check if the det_blocks function works
for i in range(0, 200) :
  X_cur, Y_cur, X_prev, Y_prev = det_blocks(x0, x1, y0, y1, w, h, a, b, X_cur, Y_cur, X_prev, Y_prev)
  print(X_cur, Y_cur, X_prev, Y_prev)
  if X_cur == 100 or Y_cur == 100 :
    break