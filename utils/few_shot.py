import sys
import random

fin = open(sys.argv[1])
test = open('text2/test.txt', 'w')
train = open('text2/train.txt', 'w')
val = open('text2/val.txt', 'w')

for line in fin:
    line = line.strip()
    sp = line.split('~')
    if len(sp) < 5:
        continue
    leader = sp[:1]
    sp = sp[1:]
    random.shuffle(sp)

    cnt = len(sp)//4
    train.write('~'.join(leader+sp[:3*cnt]))
    #val.write('~'.join(leader+sp[2*cnt:3*cnt]))
    test.write('~'.join(leader+sp[3*cnt:]))

    train.write('\n')
    val.write('\n')
    test.write('\n')
    


