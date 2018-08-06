import requests
from lxml import etree

headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'}
for i in range(501):

    base_url='http://xm.99166.com/mzdq/{}.html'.format(str(i+1))
    boy_url='http://xm.99166.com/mzdq/nan{}.html'.format(str(i+1))
    girl_url='http://xm.99166.com/mzdq/nv{}.html'.format(str(i+1))
    urls=[base_url,boy_url,girl_url]
    for url in urls:
        print("正在爬取：",url)
        res=requests.get(url,headers=headers)
        res.encoding='gbk'
        html=etree.HTML(res.text)
        # names=html.xpath('//div[@class="wrapper"]/div[@class="pagebody"]/div[@class="pleft"]/div[@class="pleftmain"]/div[@class="nameall margintop10"]/div[@class="nacon"]/ul/li/a/text()')
        names=html.xpath('//div[@class="pleftmain"]/div[@class="nameall margintop10"]/div[@class="nacon"]/ul/li/a/text()')
        new_names=[name.replace('·','').strip() for name in names]
        with open('names.txt','a',encoding='utf-8') as out_data:
            for name in new_names:
                out_data.write(name+'\n')
        res.close()