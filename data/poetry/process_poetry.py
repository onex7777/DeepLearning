import json
import glob

json_files = glob.glob('poetry_json/*.json')

with open('train.txt', 'w',encoding='utf-8') as w, open('valid.txt', 'w',encoding='utf-8') as v,\
        open('total.txt', 'w',encoding='utf-8') as to:
    for file in json_files:
        with open(file, 'r',encoding='utf-8-sig') as f:
            str_data = f.read()
            dict_data = json.loads(str_data)
            for i in range(0,int(len(dict_data)*0.9)):
                content = ''
                for j in range(len(dict_data[i]['paragraphs'])):
                    content += dict_data[i]['paragraphs'][j]
                w.write(content+'\n')
                to.write(content+'\n')
            for i in range(int(len(dict_data) * 0.9),len(dict_data)):
                content = ''
                for j in range(len(dict_data[i]['paragraphs'])):
                    content += dict_data[i]['paragraphs'][j]
                v.write(content + '\n')
                to.write(content + '\n')
