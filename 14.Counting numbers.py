num_counter = {}
#{1:1,2:3,3:2,...}
#for i in 11:

L1 = [1, 2, 2, 3, 2, 3, 4, 5]
for i in L1:
  if i in num_counter:
    num_counter[i] += 1
  else:
    num_counter[i] = 1

temp = 0
final_result = None
for K,V in num_counter.items():
  print(f"K:{K},V:{V},temp:{temp}")
  if V>temp:
    temp = V
    final_result = K



