from werkzeug.utils import secure_filename
import os
import glob
import pandas as pd
import numpy as np
import json
from pathlib import Path  # for accessing operating system path
from bs4 import BeautifulSoup
from requests_html import HTML
from requests_html import HTMLSession
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import re
from googlesearch import search
from datetime import date
import time



class Scrapper():

    def __init__(self,prod_link_path,retailers_path,aggregated_data_path,log_path,keywords):
        self.prod_link_path=prod_link_path
        self.retailers_path=retailers_path
        self.aggregated_data_path=aggregated_data_path
        self.log_path=log_path
        self.keywords=keywords
        self.retailer=""
        self.retaler_URL=""
        self.desc_id=""

    def Process_Data(self,prod_link_path,keywords):
        def strip(text):
            try:
                return text.strip()
            except AttributeError:
                return text

        data = pd.read_csv(prod_link_path,
                            index_col=None,
                            converters = {'calumet_product' : strip,
                                    'competitor_product' : strip})

        return data

    def GetPriceWalmart(self,main_prod):
        try:
            retailer=self.retailer
            prod=re.sub('[^a-zA-Z0-9 \n\.]', ' ', main_prod)
            domain= self.retailer_URL
            URL = domain + "/search?q=" + prod.replace(" ","+")  
            print(URL)
            desc_id = self.desc_id         
            # # Read the search result page
            req = Request(URL , headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
            webpage = urlopen(req).read()
            time.sleep(10) # wait for 10 seconds to avoid getting blocked
            page_soup = BeautifulSoup(webpage, "html.parser")
            print(page_soup)
            page_urls = page_soup.findAll(name='a')
            page_urls = [img.get('href') for img in page_urls if img.get('href').startswith("/ip/")][0]

            req = Request(domain + page_urls , headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
            webpage = urlopen(req).read()
            page_soup = BeautifulSoup(webpage, "html.parser")
            print(page_soup)
            desc = page_soup.find("div",class_="mr3 ml7 self-center")
            price = str(page_soup.find("span",itemprop="price").text)
            print("Fetched details for {}".format(main_prod))
        except:
            price=0
            print("Couldn't fetch details for {}".format(main_prod))

        return price

    def GetPrice(self,ob):
        if self.retailer=="Walmart":
            price= self.GetPriceWalmart(ob['competitor_product'])
        return np.random.randint(0,10)

    def ScrapperLogs(self,run_date,run_time,status):
        # read the log file
        cols=['Retailer','Date','RunTime(mins)','Status']
        temp_log=pd.DataFrame([[self.retailer,run_date,run_time,status]],columns=cols)

        if os.path.exists(self.log_path):
            log_data = pd.read_csv(self.log_path,index_col=None)
        else:
            log_data = pd.DataFrame(columns=cols)

        log_data = log_data.append(temp_log,ignore_index=True)
        log_data.to_csv(self.log_path,index=False)

        return None


def main():
    basedir = os.path.abspath(os.path.dirname(__file__))

    prod_link_path = os.path.join(basedir, 'retail_product_link_data.csv')
    retailers_path = os.path.join(basedir, 'retailers.csv')
    aggregated_data_path= os.path.join(basedir,'aggregated_price_data.csv')
    log_path = os.path.join(basedir,'main_log.csv')
    keywords = {"trans":"transmission","perf":"performance","min":"mineral","eng":"engine","syn":"synthetic","rac":"racing","est":"ester","Ounce": "oz"}

    scrapper = Scrapper(prod_link_path,retailers_path,aggregated_data_path,log_path,keywords)
    data = scrapper.Process_Data(prod_link_path, keywords)

    data = data.loc[data['competitor_product']=="Berryman B-12 Chemtool 15 Ounce Carburetor, Fuel system And Injector Cleaner"]

    retailers_data = pd.read_csv(retailers_path,index_col=None)
    
    for retailer in retailers_data['retailer_name']:
        start_time=time.time()
        scrapper.retailer=retailer
        scrapper.retailer_URL = retailers_data.loc[retailers_data['retailer_name']==scrapper.retailer]['URL'].values[0]
        scrapper.desc_id = retailers_data.loc[retailers_data['retailer_name']==scrapper.retailer]['desc_id'].values[0]
        temp_data=data.copy()
        temp_data['price'] = temp_data.apply(scrapper.GetPrice,axis=1)
        temp_data['datetime'] = pd.to_datetime('today').strftime("%m/%d/%Y, %H:%M:%S")

        agg_cols = ['calumet_product','competitor_product','price','datetime']

        fail_count = len(temp_data.loc[temp_data['price']==0])
        pass_count = len(temp_data)-fail_count

        # read the aggregated file
        if os.path.exists(aggregated_data_path):
            agg_data = pd.read_csv(aggregated_data_path,index_col=None)
        else:
            agg_data = pd.DataFrame(columns=agg_cols)
        

        agg_data = agg_data.append(temp_data,ignore_index=True)
        agg_data['retailer']=retailer

        agg_data.to_csv(aggregated_data_path,index=False)
        
        run_time = time.time() - start_time
        status="Fetched data for all products" if fail_count==0 else "Passed for {} products, Failed for {} products".format(pass_count,fail_count)
        scrapper.ScrapperLogs(pd.to_datetime('today').strftime("%m/%d/%Y, %H:%M:%S"),run_time,status)

if __name__=="__main__":
    main()
