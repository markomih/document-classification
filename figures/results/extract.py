data = None
import re

def get_header(file_path):
    header = None
    with open(file_path) as f: 
        header = f.readline().strip()
    
    header = re.sub(r'=([a-z]|[A-Z]|[0-9]|\.)*( |)', ' ', header).strip()
    header=header.replace(' ', ',')
    return header

def get_csv_content(file_path):
    content=[]
    with open(file_path) as f: 
        for line in f:
            args = line.strip().split(' ')
            for i in range(len(args)):
                args[i] = args[i][args[i].find('=')+1:]
            
            content = content + [','.join(args)]
    
    content_str = '\n'.join(content)
    print(content_str)
    return content_str

if __name__ == '__main__':
    h = get_header('log.txt')
    b = get_csv_content('log.txt')

    with open('out.csv', 'w') as f: 
        f.write('\n'.join([h, b]))