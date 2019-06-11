f = open('text/metabolites.xml')

trig = False
syn = False
names = []
for line in f:
    if '</secondary_accessions>' in line:
        trig = True
    if trig and '<name>' in line:
        name_s = line.find('<name>') + len('<name>')
        name_e = line.find('</name>')
        names.append(line[name_s:name_e])
        trig = False
        syn = True
    if syn and '</synonyms>' in line:
        syn = False
    if syn and '<synonym>' in line:
        name_s = line.find('<synonym>') + len('<synonym>')
        name_e = line.find('</synonym>')
        names.append(line[name_s:name_e])

w = open('text/metabolites.txt', 'w')
w.write('\n'.join(names))
w.close()


