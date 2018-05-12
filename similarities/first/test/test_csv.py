with open('../run/log.txt','r') as f:
    lines = f.read().split('\n')

    words = [word.strip() for word in lines[0].split(' ') if word.strip() != '']
    parameters = [word[:word.find('=')] for word in words]

    values = []
    for line in lines:
        words = [word.strip() for word in line.split(' ') if word.strip() != '']
        tmp = [word[word.find('=')+1:].strip() for word in words]
        values.append(','.join(tmp))

    to_save = ','.join(parameters) + '\n' + '\n'.join(values)
    with open('log.csv', 'w') as s:
        s.write(to_save)
