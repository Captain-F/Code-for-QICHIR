import requests
import os
import re
import random
import time
import pandas as pd
from tqdm import tqdm



def get_images_from_baidu(keyword, kdx):
    # Due to the image copyright, the code for image scraping is presented;
    HEADERS_LIST = [
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
        "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
        "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
        "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
        "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
        "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
        "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
    ]
    header = {'User-Agent': random.choice(HEADERS_LIST)}
    url = 'https://image.baidu.com/search/acjson?'
    num = get_pic_num(keyword)
    if num <= 10000000:
        for pn in range(0, 800, 20):
            print('*' * 15 + 'page: {}'.format(int(pn/30)) + '*' * 15)
            param = {'tn': 'resultjson_com',
                     # 'logid': '7603311155072595725',
                     'ipn': 'rj',
                     'ct': 201326592,
                     'is': '',
                     'fp': 'result',
                     'queryWord': keyword,
                     'cl': 2,
                     'lm': -1,
                     'ie': 'utf-8',
                     'oe': 'utf-8',
                     'adpicid': '',
                     'st': -1,
                     'z': '',
                     'ic': '',
                     'hd': '',
                     'latest': '',
                     'copyright': '',
                     'word': keyword,
                     's': '',
                     'se': '',
                     'tab': '',
                     'width': '',
                     'height': '',
                     'face': 0,
                     'istype': 2,
                     'qc': '',
                     'nc': '1',
                     'fr': '',
                     'expermode': '',
                     'force': '',
                     'cg': '',    # 这个参数没公开，但是不可少
                     'pn': pn,    # 显示：30-60-90
                     'rn': '30',  # 每页显示 30 条
                     'gsm': '1e',
                     '1618827096642': ''
                     }
            try:
                request = requests.get(url=url, headers=header, params=param)
                if request.status_code == 200:
                    print('Request successful...')
                request.encoding = 'utf-8'
                # 正则方式提取图片链接
                html = request.text
                img_url_lists = re.findall('"thumbURL":"(.*?)",', html, re.S)
                captions = re.findall('"fromPageTitleEnc":"(.*?)",', html, re.S)
                #print(image_url_list)

                if not os.path.exists('nh_imgs/' + str(kdx) + '/imgs/'):
                    os.makedirs('nh_imgs/' + str(kdx) + '/imgs/')
                for idx, img_url in enumerate(img_url_lists):
                    img_data = requests.get(url=img_url, headers=header).content
                    with open('nh_imgs/' + str(kdx) + '/imgs/' + '{}.jpg'.format(idx + pn), 'wb') as f:
                        f.write(img_data)


                time.sleep(5)
                print('{} page written...'.format(int(pn/30)))
            except:
                continue
    else:
        print(num)


def get_pic_num(keyword):
    HEADERS_LIST = [
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
        "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
        "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
        "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
        "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
        "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
        "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
    ]
    header = {'User-Agent': random.choice(HEADERS_LIST)}
    url = 'https://image.baidu.com/search/acjson?'

    param = {'tn': 'resultjson_com',
             # 'logid': '7603311155072595725',
             'ipn': 'rj',
             'ct': 201326592,
             'is': '',
             'fp': 'result',
             'queryWord': keyword,
             'cl': 2,
             'lm': -1,
             'ie': 'utf-8',
             'oe': 'utf-8',
             'adpicid': '',
             'st': -1,
             'z': '',
             'ic': '',
             'hd': '',
             'latest': '',
             'copyright': '',
             'word': keyword,
             's': '',
             'se': '',
             'tab': '',
             'width': '',
             'height': '',
             'face': 0,
             'istype': 2,
             'qc': '',
             'nc': '1',
             'fr': '',
             'expermode': '',
             'force': '',
             'cg': '',  # 这个参数没公开，但是不可少
             'pn': 30,  # 显示：30-60-90
             'rn': '30',  # 每页显示 30 条
             'gsm': '1e',
             '1618827096642': ''
             }
    request = requests.get(url=url, headers=header, params=param)
    if request.status_code == 200:
        print('Request successful...')
    request.encoding = 'utf-8'
    html = request.text
    num = re.findall('"displayNum":\d*', html, re.S)[0].split(':')[1]
    return int(num)


if __name__ == '__main__':

    intangible_names = ["桃花坞木板年画", "杨柳青木板年画", "朱仙镇木板年画"]
    #intangible_names = ["北京兔儿爷", "大吴泥塑", "惠山泥人", "泥人张"]
    #intangible_names = ["苏绣", "蜀绣", "湘绣", "粤绣"]

    for i in tqdm(range(len(intangible_names))):
        keyword, kdx = intangible_names[i], i
        print(keyword, kdx)
        get_images_from_baidu(keyword, kdx)




