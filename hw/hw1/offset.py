input = 100
size = 12

for rank in range(size):
  len = (int)(input/size)
  offset = len * rank
  partner_len = (int)(input/size)
  if rank - 1 >=0 and rank - 1 < input%size:
    partner_len += 1
  if rank < input%size: 
    len += 1
  if rank-1 < input%size:
    offset += rank
  elif rank-1 >= input%size:
    offset += input%size
  print("rank = ", rank, "offset = ", offset, "len =", len, "partner_len = ", partner_len)
