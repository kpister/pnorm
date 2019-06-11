total = 50
ones = 0
twos = 0
wrong = 0
for line in open('preds2.csv'):
    val = line.split(',')[0]
    if val == 'x':
        wrong += 1
    if val == '1':
        ones += 1
    if val == '2':
        twos += 1
    total -= 1
    if total < 0:
        break

total = 50

print(f'Total patents analyzed:\t{total}')
print(f'Correct in first cluster:\t{ones}\t{ones/total * 100:.1f}%')
print(f'Correct in second cluster:\t{twos}\t{twos/total * 100:.1f}%')
