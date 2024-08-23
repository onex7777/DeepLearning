# encoding:UTF-8
import csv
import json
import requests
from lxml import etree
from bs4 import BeautifulSoup
import pandas as pd
import json
# https://movie.douban.com/top250?start={}&filter=

class DouBan:
    def __init__(self):
        self.temp_url = "https://movie.douban.com/top250?start={}&filter="
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36"}

    def get_url_list(self):
        return [self.temp_url.format((i-1)*25) for i in range(1, 11)]

    def parse_url(self, url):
        response = requests.get(url, headers=self.headers)
        return response.content.decode()

    def get_content_list(self, html_str,content_list):
        html = etree.HTML(html_str)
        li_list = html.xpath("//*[@id='content']/div/div[1]/ol/li")
        for li in li_list:
            item = {}
            item["rank"] = li.xpath("div/div[1]/em/text()")[0]
            item["名字"] = li.xpath("div/div[2]/div[1]/a/span[1]/text()")[0]
            item["评价人数"] = li.xpath("div/div[2]/div[2]/div/span[4]/text()")[0]
            item_temp = li.xpath("div / div[2] / div[2] / p[1] / text()[2]")[0].strip()
            item["类别"] = item_temp.replace("\xa0","")
            content_list.append(item)
        return content_list

    def save_content_list(self, content_list):
        file_name = 'data.json'
        with open(file_name, 'w',encoding='utf-8') as file_object:
            json.dump(content_list, file_object,ensure_ascii=False,indent=4)
        for content in content_list:
            print(content)

    def run(self):
        # 1 获取URL列表
        url_list = self.get_url_list()
        # 2 遍历发送请求 获取响应
        content_list = []
        for url in url_list:
            html_str = self.parse_url(url)
            # 3 提取数据
            content_list = self.get_content_list(html_str, content_list)
        # 保存
        self.save_content_list(content_list)


def trans(jsonpath, csvpath):
    json_file = open(jsonpath, 'r', encoding='utf8')
    csv_file = open(csvpath, 'w', newline='',encoding='utf8')
    keys = []
    writer = csv.writer(csv_file)

    json_data = json_file.read()
    dic_data = json.loads(json_data, encoding='utf8')

    for dic in dic_data:
        keys = dic.keys()
        # 写入列名
        writer.writerow(keys)
        break

    for dic in dic_data:
        for key in keys:
            if key not in dic:
                dic[key] = ''
        writer.writerow(dic.values())
    json_file.close()
    csv_file.close()


if __name__ == '__main__':
    douban = DouBan()
    douban.run()
    trans('data.json', 'data.csv')
