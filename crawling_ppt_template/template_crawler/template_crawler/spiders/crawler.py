import scrapy
from bs4 import BeautifulSoup
from scrapy import Request
import requests
from scrapy.linkextractors import LinkExtractor
import os
import re
import wget
import csv

dirname = os.path.dirname(os.path.abspath(__file__))

data_list = []


def get_seed_urls():
    url = 'https://www.free-power-point-templates.com/categories/#content'
    seed_url = []
    page = requests.get(url)
    soup = BeautifulSoup(page.content,'html.parser').find_all('a',href = True)
    for link_soup in soup:
        link = link_soup['href']
        match = re.search('/category/',link)
        if match is not None:
            seed_url.append(link)
    return seed_url


def download_template(url,category):

    html_doc = requests.get(url)
    print(url)
    print(html_doc)
    soup = BeautifulSoup(html_doc.content, 'html.parser').find_all('a', {'class': 'post__btn', 'href': True})
    links = [href_obj['href'] for href_obj in soup]
    next_page_url_soup = BeautifulSoup(html_doc.content, 'html.parser').find('a', {'class': 'nextpostslink','href': True})
    if next_page_url_soup is not None:
        next_page_url = next_page_url_soup['href']
    else:
        next_page_url = None
    print('Next Page:',next_page_url)

    download_prefix_link = 'https://www.free-power-point-templates.com/wp-content/files/'
    for trial_link in links:
        download_page = BeautifulSoup(requests.get(trial_link).content, 'html.parser').find_all('a', {'id': 'download'})
        for download_link in download_page:
            name = download_link.text
            #m = re.search('.pptx', name)
            if re.search('.pptx',name) or re.search('.zip',name):
                filename = wget.download(download_prefix_link + name.split()[-1],out = os.path.join(dirname,category))
                print(name)

    return next_page_url

def extract_description(url,category):
    html_doc = requests.get(url)
    ppt_url = []
    ppt_link_soup = BeautifulSoup(html_doc.content, 'html.parser').find_all('h3',{'class':'post__title'})
    for s in ppt_link_soup:
        ppt_url.append(s.a['href'])

    next_page_url_soup = BeautifulSoup(html_doc.content, 'html.parser').find('a',
                                                                             {'class': 'nextpostslink', 'href': True})
    if next_page_url_soup is not None:
        next_page_url = next_page_url_soup['href']
    else:
        next_page_url = None

    print(next_page_url)

    content_list = []

    for pp_url in ppt_url:
        depth_1_doc = requests.get(pp_url)
        text_soup = BeautifulSoup(depth_1_doc.content, 'html.parser').find('div',{'class':'entry__content'})
        p_soup = text_soup.find_all('p',{'class':None})
        content = ''
        for p_tags in p_soup:
            if content == '':
                content = p_tags.text
            else:
                content = content + ' ' + p_tags.text
        content_list.append(content)

    soup = BeautifulSoup(html_doc.content, 'html.parser').find_all('a', {'class': 'post__btn', 'href': True})
    links = [href_obj['href'] for href_obj in soup]
    name_list = []

    for trial_link in links:
        download_page = BeautifulSoup(requests.get(trial_link).content, 'html.parser').find_all('a',
                                                                                                {'id': 'download'})
        for download_link in download_page:
            name = download_link.text
            if re.search('.pptx', name) or re.search('.zip', name):
                print(name)
                name = name.split()[-1]
                name = name.replace('-template-16x9.', '.')
                name = name.replace('_ppt', '.')
                name = name.replace('.ppt','')
                name = name.replace('.pptx','')
                name = name.replace('.zip','')
                name_list.append(name)

    d = list(zip(name_list,content_list))
    data_list.extend(d)
    with open('ppt_description.csv','a') as file_obj:
        writer = csv.writer(file_obj, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for name, desc in d:
            writer.writerow([name,desc])

    print('name list',len(name_list))
    print('content_list',len(content_list))
    print('data list is:',data_list)
    print('content is:',content)
    print('Next page url is:',next_page_url)
    return next_page_url


class temp_crawler(scrapy.Spider):
    name = 'crawler'
    allowed_domains = ['free-power-point-templates.com']
    custom_settings = {
        # 'SCHEDULER_DISK_QUEUE': 'scrapy.squeues.PickleFifoDiskQueue',
        # 'SCHEDULER_MEMORY_QUEUE': 'scrapy.squeues.FifoMemoryQueue',
        # 'DUPEFILTER_CLASS': 'custom_filter.SeenURLFilter',
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.RFPDupeFilter',
        'CONCURRENT_REQUESTS': 50,
        'RETRY_ENABLED': False,
        'DOWNLOAD_TIMEOUT': 15,
        'REDIRECT_ENABLED': False,
        #'DEPTH_LIMIT': 2
    }
    #seed_url = get_seed_urls()
    seed_url = ['https://www.free-power-point-templates.com/category/transportation/',
                'https://www.free-power-point-templates.com/category/ppt-by-topics/music/'
                ]
    category = ''
    #print(seed_url[:10])
    #print(len(seed_url))

    # def start_requests(self):
    #     for i,url in self.seed_url:
    #         self.category = url.split('/')[-1]
    #         os.mkdir(self.category)
    #         yield scrapy.Request(url, callback = self.download_ppt)

    def start_requests(self):
        for url in self.seed_url:
            print(url)
            self.category = url.split('/')[-2]
            print(self.category)
            #print(url.split('/'))
            if not os.path.exists(os.path.join(dirname,self.category)):
                os.mkdir(os.path.join(dirname,self.category))
            d = {'category':self.category}
            yield scrapy.Request(url, callback = self.download_ppt,meta = d)

    def download_ppt(self,response):

        if response.status == 200:
            print(response.url,'Depth:',response.meta)
            categ = response.meta['category']

            #next_page_url = download_template(response.url,categ)
            next_page_url = extract_description(response.url,categ)

            if next_page_url is not None:
                print('follow')
                yield response.follow(next_page_url, self.download_ppt, meta = response.meta)



