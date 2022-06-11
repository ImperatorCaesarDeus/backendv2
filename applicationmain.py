from flask import Flask, render_template,request, send_from_directory,send_file,url_for,redirect,send_from_directory
from flask.helpers import safe_join
from werkzeug.utils import secure_filename
import os
import glob
import pandas as pd
import numpy as np
import json
import random
import matplotlib
from scrapper.retail_scrapper import Scrapper
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # for notebook plotting
from pathlib import Path  # for accessing operating system path
from itertools import islice
import matplotlib.dates as mdates
from datetime import date
from datetime import datetime,timedelta
from math import sqrt
import sys
from flask_socketio import SocketIO, emit
from skimage.metrics import structural_similarity as ssim
import cv2
import requests
from bs4 import BeautifulSoup
from requests_html import HTML
from requests_html import HTMLSession
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import re
from googlesearch import search
import networkx as nx
from pyvis.network import Network
import html
import plotly
import plotly.express as px
from datetime import date
import time
import chart_studio.plotly as py
import plotly.graph_objs as go
from prophet import Prophet
from flask_dropzone import Dropzone

from flask import Flask
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from curses.ascii import SUB
from flask_wtf import FlaskForm
from wtforms import StringField,  PasswordField, SubmitField
from wtforms.validators import Length, EqualTo, Email, DataRequired, ValidationError
from curses import flash
from flask import render_template, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#from appmain.scrapper.models import Item, User  

application = Flask(__name__)
socketio = SocketIO(application,cors_allowed_orid2gins="*",async_mode="threading")
keywords = {"trans":"transmission","perf":"performance","min":"mineral","eng":"engine","syn":"synthetic","rac":"racing","est":"ester",}
#flask constructor
#server configuration with python

upload_error = []
data_file_name =""
basedir = os.path.abspath(os.path.dirname(__file__))
main_data = pd.DataFrame()
filter_list=""
filter_by=""
pred_len=0
epochs=0
fvar=""
status=0

application.config.update(
    SQLALCHEMY_DATABASE_URI = 'sqlite:///userdata.db', #URI uniform resource identifier
    SECRET_KEY = '8c3ac9290e1e1affc1e6cad5',
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    PLOT_DATA_PATH=os.path.join(basedir, 'static/plots'),
    DATA_PATH=os.path.join(basedir, 'data/'),
    SCRAPPER_PATH=os.path.join(basedir,'scrapper/'),
    DROPZONE_ALLOWED_FILE_CUSTOM=True,
    DROPZONE_ALLOWED_FILE_TYPE='.csv',
    DROPZONE_MAX_FILE_SIZE=10,
    DROPZONE_MAX_FILES=2,
    DROPZONE_IN_FORM=True,
    DROPZONE_UPLOAD_ON_CLICK=True
)
dropzone = Dropzone(application)

from flask_login import UserMixin

db = SQLAlchemy(application)
bcrypt = Bcrypt(application)
login_manager = LoginManager(application)
login_manager.login_view = "login_page"
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(length=30), nullable=False, unique=True)
    email_address = db.Column(db.String(length=50), nullable=False, unique=True)
    password_hash = db.Column(db.String(length =60), nullable=False)
    items = db.relationship('Item', backref='owned_user', lazy=True) 
    #backref to user model. allows to share user with what is in Item model.
    #lazy = true makes sqlalchemy grab all the items in one shot.

    @property
    def password(self):
        return self.password

    @password.setter #sets pasword to user instance
    def password(self, plain_text_password):
        self.password_hash = bcrypt.generate_password_hash(plain_text_password).decode('utf-8')
    
    def check_password_correction(self, attempted_password):
        return bcrypt.check_password_hash(self.password_hash, attempted_password)


class Item(db.Model):
    id = db.Column(db.Integer(), primary_key=True) # unique identifier for the flask model
    name = db.Column(db.String(length=30),nullable=False,unique=True)
    datecompleted = db.Column(db.Integer(), nullable=False)
    description =db.Column(db.String(length=2048),nullable=False, unique=True)
    owner = db.Column(db.Integer(), db.ForeignKey('user.id')) #foreignkey searches for the primary key
    #db.ForeignKey sets the relation between the Item and User classes.
    def __repr__(self):
        return f'Item {self.name}'

def Process_Data(prod_link_path,keywords):
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

        #called from the web scraping function below.
        #data = ... reads the csv from the given path and calls the strip function which gets the data from the string

def GetPriceWalmart2(main_prod,retailer,retailer_URL,desc_id):
    try:
        retailer=retailer
        prod=re.sub('[^a-zA-Z0-9 \n\.]', ' ', main_prod)
        #re.sub removes anything that is not a letter or number!
        domain= retailer_URL
        URL = domain + "/search?q=" + prod.replace(" ","+")
        #configures the URL. .replace replaces blank " " with +'s
        desc_id = desc_id            
        # Read the search result page
        req = Request(URL , headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
        # request allows for HTTP requests to be sent with ease therefore accessing a resource.
        webpage = urlopen(req).read()
        #opens webpage
        waittimeran = random.randint(5,20)
        time.sleep(waittimeran) # wait for 10 seconds to avoid getting blocked
        page_soup = BeautifulSoup(webpage, "html.parser")
        #literally parses HTML of the webpage.
        page_urls = page_soup.findAll(name='a')
        #print('page urls output with product names: ')
        #print(page_urls)
        #dont know why it wants 'a' # finds all a locations in the html to find the searchbar
        
        page_urls = [img.get('href') for img in page_urls if img.get('href').startswith("/ip/")]
        print('the length of the list is: ', len(page_urls))
        #print('all page url results: ')
        #print(page_urls)
        pgulen = len(page_urls)
        iplist = []
        for x in range(pgulen):
            page_urls = [img.get('href') for img in page_urls if img.get('href').startswith("/ip/")][x]
            iplist.append(page_urls)
            print([img.get('href') for img in page_urls if img.get('href').startswith("/ip/")][x])
        
        vectorizer = CountVectorizer()
        features = vectorizer.fit_transform(iplist).todense()

        x = len(iplist)

        lst = []

        for f in features:
            lst.append(euclidean_distances(features[0], f))
            print(euclidean_distances(features[0], f))

        lst.remove(0.)

        count = 1
        for x in lst:
            count = count + 1

        minium = min(lst)

        numpos = lst.index(minium)
        numpos = numpos + 1

        print('the product to find is: ', iplist[0])
        print('the matching product is: ', iplist[numpos])

        #print('page urls selecting product: ')
        #print(page_urls)
        #finds image locations in HTML
        req = Request(domain + iplist[numpos] , headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
        #requests to access the image/images and store them in req
        waittimeran2 = random.randint(5,20)
        time.sleep(waittimeran2)
        webpage = urlopen(req).read()
        page_soup = BeautifulSoup(webpage, "html.parser")
        #desc = page_soup.find("div",class_="mr3 ml7 self-center")#not sure why this is here
        price = str(page_soup.find("span",itemprop="price").text)
        #print("Fetched details for {}".format(main_prod))
        #print("The price for {}".format(main_prod))
        print(price)
    except:
        price=0
        #print("Couldn't fetch details for {}".format(main_prod))
        #print(page_urls)
        #print(page_soup)
    return price

    #function is called from the getprice2 function.
    #it gets the product images and then scrapes the price listed from the product and returns it.


def GetPrice2(ob):
    retailer="Walmart"
    if retailer=="Walmart":
        retailer_URL="https://www.walmart.com"
        desc_id="mb1 ph1 pa0-xl bb b--near-white w-25"
        price= GetPriceWalmart2(ob['competitor_product'],retailer,retailer_URL,desc_id)
        #inputs arguments into getpricewalmart2 and returns the price to the start_scrapper() function
    return price

def GetPrice3(ob):
    retailer="Walmart"
    if retailer=="Walmart":
        retailer_URL="https://www.walmart.com"
        desc_id="mb1 ph1 pa0-xl bb b--near-white w-25"
        price= GetPriceWalmart2(ob['calumet_product'],retailer,retailer_URL,desc_id)
    return price

def ScrapperLogs(log_path,retailer,run_date,run_time,status):
    # read the log file
    cols=['Retailer','Date','RunTime(mins)','Status']
    temp_log=pd.DataFrame([[retailer,run_date,run_time,status]],columns=cols)

    if os.path.exists(log_path): #checks if the log_path exists 
        log_data = pd.read_csv(log_path,index_col=None)
    else:
        log_data = pd.DataFrame(columns=cols)

    log_data = log_data.append(temp_log,ignore_index=True)
    log_data.to_csv(log_path,index=False)

    return None
    #this function logs the scrapper details from when it runs(retailer, date,runtime,status/results).

@application.route('/upload', methods=['POST'])
@login_required
def handle_upload():
    """
    uploads the selected files on the server.
    :return: training parameter selection html
    """
    global upload_error
    for key, f in request.files.items():
        try:
            #file_error = UploadError()
            if key.startswith('file'):
                filename = secure_filename(f)
                print(filename)
                f.save(os.path.join(application.config['SCRAPPER_PATH'], filename))
        except Exception as e:
            #file_error.file_name = f.filename
            #file_error.error = e.message
            #upload_error.append(file_error)
            pass

    return redirect(url_for('scrapper_log'))





@application.route('/download/<filename>')
@login_required
def download(filename):
    if filename=="templates_comp":
        return send_from_directory('static','retail_product_link_data.csv')
    elif filename=="templates_ret":
        return send_from_directory('static','retailers.csv')
    elif filename=="logs":
        return send_from_directory(application.config['SCRAPPER_PATH'],'main_log.csv')
    elif filename=="agg_data":
        return send_from_directory(application.config['SCRAPPER_PATH'],'aggregated_price_data.csv')

#

#This function is not called!

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    return err

##########################

#Function is called in compare images below

def get_img_from_url(img):
    req = Request(img , headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
    req = urlopen(req).read()
    arr = np.asarray(bytearray(req), dtype=np.uint8)
    imgarr = cv2.imdecode(arr, -1)
    grayImg = cv2.cvtColor(imgarr, cv2.COLOR_BGR2GRAY) if len(imgarr.shape)==3 else imgarr
    return grayImg

    #converts image to greyscale for processing

def process_image(img):
    gray = 255*(img < 128).astype(np.uint8) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
    return rect


    #function is called in getDistributors and GetPriceWalmart args are self explanatory 

def compare_images(base_img_url, product_img_url):
    try:
        imageA = get_img_from_url(base_img_url)
        imageB = get_img_from_url(product_img_url)
        imageA = process_image(imageA) #conv to grey
        imageB = process_image(imageB) #conv to grey
        if imageA.shape>imageB.shape:
            imageA = cv2.resize(imageA, (imageB.shape[1],imageB.shape[0]), interpolation = cv2.INTER_AREA)
        else:
            imageB = cv2.resize(imageB, (imageA.shape[1],imageA.shape[0]), interpolation = cv2.INTER_AREA)
        
        #resize for ssim

        s = ssim(imageA, imageB) #image structural similary index
    except:
        s=0
    return s

    #So this function returns the similiarity of two images.

#This function is not called anywhere!

def search_web():
    URL="https://www.belray.com/?s=gear+saver+transmission+oil"
    #URL= "https://nms.kcl.ac.uk/rll/index.html"
    #page=requests.get(URL)
    #soup = BeautifulSoup(page.content,"html.parser")
    #results = soup.find("div",class_="posts_group classic")

    # page=requests.get(URL)
    # soup = BeautifulSoup(page.content,"html.parser")
    # print(soup)


    sel = "/html/body/div/div[1]/div/div/div/div/main/div/div[2]/div/div/div[1]/section"
    session = HTMLSession()
    response = session.get(URL)   
    print(response.html.xpath('img'))

######################################

def get_prod_catalog():
    prods = pd.read_csv((os.path.join(application.config['DATA_PATH'], 'prod_list.csv')))
    return prods

def get_brand_data():
    brand_data = pd.read_csv((os.path.join(application.config['DATA_PATH'], 'brand_data.csv')),index_col=None)
    return brand_data

def get_ret_prod_catalog():
    prods = pd.read_csv((os.path.join(application.config['DATA_PATH'], 'ret_prod_list.csv')))
    return prods


def get_ret_retailer():
    ret_retailers = pd.read_csv((os.path.join(application.config['DATA_PATH'], 'ret_retailers.csv')),index_col=None)
    return ret_retailers

    #These functions collect the data from the .csv's and returns the dataframes to where the function was called.

#This function is not called!

def get_industrial_product_data(search_term):
    #URL="https://www.walmart.com/search?q=Duralec+Super+15W40+Gallons"
    #URL="https://www.belray.com/?s=gear+saver+transmission+oil"
    URL = "https://www.belray.com/?s=Foam+Filter+Oil"
    req = Request(URL , headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    page_soup = BeautifulSoup(webpage, "html.parser")
    #results = page_soup.find("div",class_="posts_group classic")
    results = page_soup.find_all("a")
    f=[]
    for r in results:
        f.append(r.get('href'))
    print(f)

################################

#This function is not called!

def get_domain(url):
    domain = urlparse(url).netloc
    return domain

###############################


class RegisterForm(FlaskForm):
    def validate_username(self, username_to_check):
        user = User.query.filter_by(username=username_to_check.data).first()
        if user:
            raise ValidationError('Username allready exists! Try another.')

    def validate_email_address(self, email_address_to_check):
        email_address = User.query.filter_by(email_address=email_address_to_check.data).first()
        if email_address:
            raise ValidationError('Email Address already exists! Try another')


    username = StringField(label='Username:', validators=[Length(min=2, max=30), DataRequired()])
    email_address = StringField(label='Email Address:', validators=[Email()])
    password1 = PasswordField(label='Password:', validators=[Length(min=6)])
    password2 = PasswordField(label='Confirm Password:', validators=[EqualTo('password1')])
    submit = SubmitField(label='Create Account')

class LoginForm(FlaskForm):
    username = StringField(label='User Name', validators=[DataRequired()])
    password = StringField(label='Password', validators=[DataRequired()])
    submit = SubmitField(label='Sign in')

@application.route('/register', methods=['GET','POST']) #allows the handling of post requests
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(username=form.username.data,
                            email_address=form.email_address.data,
                            password=form.password1.data)
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        #flash(f"Account created successfully! you are now logged in as {user_to_create.username}", category='success')
        #return redirect(url_for('index'))
        return render_template('index.html', form=form)
    if form.errors != {}: #If there are NOT errors from the validations!
        for err_msg in form.errors.values():
            flash(f'There was an error with creating a user: {err_msg}', category='danger')
    return render_template('register.html', form=form)

@application.route('/login',methods=['GET','POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(
                attempted_password=form.password.data
        ):
            login_user(attempted_user)
            return redirect(url_for('index'))
        else:
            flash('Username and Password dont match! Please try again.', category ='danger')
            #category='danger' makes red error
    return render_template('login.html', form=form)

@application.route('/logout')
def logout_page():
    logout_user()
    flash("you have been logged out!", category='info')
    return redirect(url_for('home_page'))

@application.route('/') #root url/homepage
@application.route('/home')
def home_page():
    return render_template('home.html')

@application.route("/welcome")
@login_required
def index():
    return render_template('index.html')

@application.route('/train') #converts regular function into a view function
@login_required
def train():
    prods = get_prod_catalog()
    ind_rets = get_brand_data()
    #get_industrial_product_data("Foam Filter Oil")
    return render_template("industrial_template.html", prods=prods['Product_Name'],ind_rets=ind_rets.loc[ind_rets['Calumet_Brand']=="Y",'Brand_Name'],ind_comp_rets=ind_rets.loc[ind_rets['Calumet_Brand']=="N",'Brand_Name'])

    #When triggered returns information inside a html format.


@socketio.on('get_dist') #converts regular function into a view function
def getDistributors(data):
    calumet_prod = data['prods']
    calumet_brand = data['rets']
    comp_brand_flag = data['radio_btn']
    num_crawl = int(data['num_crawl']) if data['num_crawl']!="" else 1 #num_crawl = data(num...) if it is not blank if it is num_crawl = 1
  
    prod_data = get_prod_catalog()
    base_img_url = prod_data.loc[prod_data['Product_Name']==calumet_prod,['Base_Img_URL']].values[0]
    emit('status', {"base_img_url":base_img_url[0],"d":""})
    
    query = calumet_prod + " +" + calumet_brand

    img_dict={}
    for URL in search(query, num_results=num_crawl):
        temp_ = []
        if comp_brand_flag=="calumet":
            req = Request(URL , headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
            webpage = urlopen(req).read()
            page_soup = BeautifulSoup(webpage, "html.parser")
            img_urls = page_soup.findAll(name='img')
            img_urls = [img.get('src') for img in img_urls]
            for a in page_soup.find_all("a" , href=True):
                if re.findall(r".+(?=jpg|png|jpeg)",a['href']): 
                # find out if the url contain jpg or png or jpeg , if not return a empty list. empty list is False
                    img_urls.append(a['href'])
            img_urls = list(set(img_urls))
            temp_ = [[img,compare_images(base_img_url[0],img)] for img in img_urls]
            img_dict[URL] = temp_        
        else:
            temp_=""
            img_dict[URL] = temp_
        
    

    # img_dict= {'https://www.ebay.co.uk/itm/333518229956': [['https://ir.ebaystatic.com/rs/v/fxxj3ttftm5ltcqnto1o4baovyl.png', 0.4618447519852569], 
    # ['https://ir.ebaystatic.com/cr/v/c1/BFCoupon_Doodle_150x30NEW.jpg', 0.3960588783102032],  ['https://i.ebayimg.com/thumbs/images/g/qcoAAOSwQvheStEN/s-l96.jpg', 0.25235984007857887], 
    # ['https://i.ebayimg.com/thumbs/images/g/rosAAOSwwlJeStBQ/s-l96.jpg', 0.26100369746968777],  ['https://i.ebayimg.com/thumbs/images/g/ZMcAAOSwsB9eStBI/s-l96.jpg', 0.2546681591588461], 
    # ['https://i.ebayimg.com/thumbs/images/g/8W0AAOSwuKVfIqCe/s-l96.jpg', 0.5243926787879984], ['https://i.ebayimg.com/thumbs/images/g/HxUAAOSwAKxgajsw/s-l96.jpg', 0.26484822316054374], 
    # ['https://p.ebaystatic.com/aw/pics/s.gif', 0], ['https://i.ebayimg.com/images/g/pkoAAOSwsbheStEY/s-l300.jpg', 0.25288417495122056], 
    # ['https://i.ebayimg.com/images/g/pkoAAOSwsbheStEY/s-l300.jpg', 0.25288417495122056], ['https://ir.ebaystatic.com/pictures/aw/pics/s.gif', 0], 
    # ['https://i.ebayimg.com/images/g/AGYAAOSwUg9aZwPC/s-l140.png', 0.38731873043504783], ['https://p.ebaystatic.com/aw/pics/s.gif', 0], 
    # ['https://ir.ebaystatic.com/pictures/aw/pics/s.gif', 0], ['', 0], ['https://p.ebaystatic.com/aw/pics/s.gif', 0], ['https://p.ebaystatic.com/aw/pics/s.gif', 0],
    #  ['https://p.ebaystatic.com/aw/pics/s.gif', 0], ['https://p.ebaystatic.com/aw/pics/s.gif', 0], ['//p.ebaystatic.com/aw/logos/logoPaypalCreditv2_157x55.png', 0], 
    #  ['https://i.ebayimg.com/thumbs/images/g/7cwAAOSwXzldTffj/s-l200.jpg', 0.7102715621640003], ['https://i.ebayimg.com/thumbs/images/g/erQAAOSwd01dTffe/s-l200.jpg', 0.7117629556636218], 
    #  ['https://i.ebayimg.com/thumbs/images/g/F5IAAOSwn~hdTfiP/s-l200.jpg', 0.7121204603517279], ['https://i.ebayimg.com/thumbs/images/g/oEUAAOSwRFldTfhJ/s-l200.jpg', 0.7007298669421226], 
    #  ['https://i.ebayimg.com/thumbs/images/g/wCQAAOSwfe5dTffQ/s-l200.jpg', 0.7180177671104382], ['https://i.ebayimg.com/thumbs/images/g/9Q8AAOSw-LldTfes/s-l200.jpg', 0.7096428346401873], 
    #  ['https://i.ebayimg.com/thumbs/images/g/DOIAAOSwy5xdTfeJ/s-l200.jpg', 0.7111884938508422], ['https://i.ebayimg.com/thumbs/images/g/rQIAAOSwRSddTfhU/s-l200.jpg', 0.7150656251320939], 
    #  ['https://i.ebayimg.com/thumbs/images/g/8-sAAOSwAzNcfjJ-/s-l200.jpg', 0.5717926037386712], ['https://i.ebayimg.com/thumbs/images/g/vrEAAOSwoKBf0foq/s-l200.jpg', 0.709651221255349], 
    #  ['https://i.ebayimg.com/thumbs/images/g/mY8AAOSwe4FdTffW/s-l200.jpg', 0.7103256799976717], ['https://i.ebayimg.com/thumbs/images/g/Cc8AAOSwcXBbSQWG/s-l200.jpg', 0.5097241012006759], 
    #  ['https://p.ebaystatic.com/aw/pics/s.gif', 0], ['https://p.ebaystatic.com/aw/pics/s.gif', 0], ['https://p.ebaystatic.com/aw/pics/s.gif', 0], ['https://p.ebaystatic.com/aw/pics/s.gif', 0], 
    #  ['//p.ebaystatic.com/aw/logos/logoPaypalCreditv2_157x55.png', 0], ['https://i.ebayimg.com/images/g/pkoAAOSwsbheStEY/s-l300.jpg', 0.25288417495122056], 
    #  ['https://ir.ebaystatic.com/pictures/aw/pics/s.gif', 0], ['https://rover.ebay.com/roversync/?site=3&stg=1&mpt=1636823270283', 0]], 
    # 'https://www.belray.com/product/gear-saver-transmission-oil/': [['https://www.belray.com/wp-content/uploads/2019/02/Bel-Ray-Logo_retina.png', 0.19920479106826666],
    #  ['https://www.belray.com/wp-content/uploads/2017/10/retina-brlogo.png', 0.1724584350549012], 
    #  ['https://www.belray.com/wp-content/uploads/2019/02/Bel-Ray-Logo_retina.png', 0.19920479106826666], 
    #  ['https://www.belray.com/wp-content/uploads/2019/02/Bel-Ray-Logo_retina.png', 0.19920479106826666], 
    #  ['https://www.belray.com/wp-content/uploads/2018/08/Thumper_Gear_Saver_80W_85_301722150160_P3528.02_1L_Front-80x80.jpg', 0.7254567593729239], 
    #  ['https://www.belray.com/wp-content/uploads/2018/08/H1_R_Racing_100_Syn_301407150160_P3508.03_1L_Front-80x80.jpg', 0.7171789045132488], 
    #  ['https://www.belray.com/wp-content/uploads/2018/08/Gear_Saver_Trans_Oil_80W_301708150160_P3524.02_1L_Front-600x499.jpg', 0.9981451107185701], 
    #  ['https://www.belray.com/wp-content/uploads/2018/08/Thumper_Racing_Works_Engine_Oil_Group_Front-300x300.jpg', 0.28479823384790376], 
    #  ['https://www.belray.com/wp-content/uploads/2018/08/MC_4T_Mineral_20W_50_301713150145_P3676.03_1L_Front-300x300.jpg', 0.7469681948251462], 
    #  ['https://www.belray.com/wp-content/uploads/2018/08/6-in-1_multipurpose_lube-300x300.jpg', 0.7030696633802545]]}
    
    if comp_brand_flag=="calumet":
        test=pd.DataFrame(columns=(['Page URL','Product Image URL','Match Score']))
        for (key, value) in img_dict.items():
            for v in value:
                if v[1]>0.60:
                    test_temp = pd.DataFrame([[key,v[0],v[1]]],columns=(['Page URL','Product Image URL','Match Score']))
                    test = test.append(test_temp)

        test['Product Image URL'] = html.escape("""<img style="margin:auto;" width=25% src=""" + test['Product Image URL'] + ">")
        test = test.sort_values(by='Match Score',ascending=False)
    else:
        test = pd.DataFrame(columns=(['Page URL']))
        test['Page URL'] = list(img_dict.keys())
        #test['Domain'] = test['Page URL'].apply(getDomain)
        #test['Page URL'] = html.escape("""<a href=""" + test['Page URL'] + ">"+test['Domain']+"""</a""")

    # desc_id = ret_data.loc[ret_data['Retailer_Name']==ret,['desc_id']].values[0]
    # var_id = ret_data.loc[ret_data['Retailer_Name']==ret,['var_id']].values[0]
    # req = Request(URL , headers={'User-Agent': 'Mozilla/5.0'})
    # webpage = urlopen(req).read()
    # page_soup = BeautifulSoup(webpage, "html.parser")
    # desc = str(page_soup.find("div",id=desc_id[0]))
    # var = str(page_soup.find("div",id=var_id[0]))

    
    # desc="""<div id="tab-description">
    #         <p>Bel-Ray Gear Saver Transmission Oil is a gear oil that has been developed for the unique demands of all motorcycle transmissions equipped with wet clutches. Bel-Ray Gear Saver Transmission Oil flows freely for better clutch cooling and provides positive clutch engagement for better starts and longer clutch life while protecting highly-loaded gears from wear, ensuring smooth shifts for more positive action.</p>
    #         <ul>
    #         <li style="list-style-type: none;">
    #         <ul>
    #         <li>Formulated to protect transmission gears from wear</li>
    #         <li>Positive clutch engagement for better starts and longer clutch life</li>
    #         <li>Protects shafts, bearings and gears</li>
    #         <li>For air-cooled/liquid-cooled 2T/4T transmissions with wet clutches</li>
    #         </ul>
    #         </li>
    #         </ul>
    #         </div>"""

    #img_dict = pd.concat({k: pd.Series(v) for k, v in img_dict.items()})

    emit('status', {"status":1,"base_img_url":base_img_url[0],"d":html.unescape(test.to_html(table_id="dist_table"))})
    return None

@application.route('/get_ind_data', methods=['POST'])
@login_required
def get_ind_prd_spec():
    prod = request.form.get("prod").lower()
    for w in prod.split(" "):
        prod = prod.replace(w,keywords.get(w)) if keywords.get(w) is not None else prod
    prod = prod.replace(" ","-")
    ret = request.form.get("ret")
    ret_data = get_industrial_retailer()
    URL = ret_data.loc[ret_data['Retailer_Name']==ret,['URL']].values[0]
    URL = URL[0] + prod + '/'
    print(URL)
    req = Request(URL , headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    page_soup = BeautifulSoup(webpage, "html.parser")
    res = page_soup.find("div",id="tab-description")
    print(res)
    return None
    
    #return render_template("index.html",res= res)



@socketio.on('get_spec')
def GetSpec(data):
    prod = data['prods'].lower()
    for w in prod.split(" "):
        prod = prod.replace(w,keywords.get(w)) if keywords.get(w) is not None else prod
    prod = prod.replace(" ","-")
    ret = data['rets']
    ret_data = get_industrial_retailer()
    URL = ret_data.loc[ret_data['Retailer_Name']==ret,['URL']].values[0]
    URL = URL[0] + prod + '/'
    desc_id = ret_data.loc[ret_data['Retailer_Name']==ret,['desc_id']].values[0]
    var_id = ret_data.loc[ret_data['Retailer_Name']==ret,['var_id']].values[0]
    req = Request(URL , headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    page_soup = BeautifulSoup(webpage, "html.parser")
    desc = str(page_soup.find("div",id=desc_id[0]))
    var = str(page_soup.find("div",id=var_id[0]))

    
    # desc="""<div id="tab-description">
    #         <p>Bel-Ray Gear Saver Transmission Oil is a gear oil that has been developed for the unique demands of all motorcycle transmissions equipped with wet clutches. Bel-Ray Gear Saver Transmission Oil flows freely for better clutch cooling and provides positive clutch engagement for better starts and longer clutch life while protecting highly-loaded gears from wear, ensuring smooth shifts for more positive action.</p>
    #         <ul>
    #         <li style="list-style-type: none;">
    #         <ul>
    #         <li>Formulated to protect transmission gears from wear</li>
    #         <li>Positive clutch engagement for better starts and longer clutch life</li>
    #         <li>Protects shafts, bearings and gears</li>
    #         <li>For air-cooled/liquid-cooled 2T/4T transmissions with wet clutches</li>
    #         </ul>
    #         </li>
    #         </ul>
    #         </div>"""
    
    emit('status', {"desc":desc,"status":1,"var":var})
    return None

################################

@socketio.on('get_price')
def GetPrice(data):
    main_prod= data['prods']
    ret = data['rets']

    if (ret=="Walmart"):
        base_img_url, final_img_dict, pr, max_score, max_key = GetPriceWalmart(main_prod)
    

    if max_score>0.20:
        status="Match found"
    else:
        status:"Match not found"
        pr="Not Applicable"
        final_img_dict[max_key]=""

    emit('status', {"desc":pr,"status":status,"base_img_url":base_img_url[0],"ret_img_url": final_img_dict[max_key],"price":pr})



@application.route('/get_ret_data') #this triggers the function
@login_required
def get_ret_prd_spec():
    ret_prods = get_ret_prod_catalog()
    ret_retailers = get_ret_retailer()
    # path = os.path.join(basedir, 'templates/test.html')
    # with open(path, 'r') as f:
    #     contents = f.read()
    #     soup = BeautifulSoup(contents, 'lxml')
    # desc = soup.find("div",class_="mb1 ph1 pa0-xl bb b--near-white w-25")
    # pr = str(desc.find("div",class_="b black f5 mr1 mr2-xl lh-copy f4-l").get_text())
   
    return render_template("retail_template.html",prods=ret_prods['Product_Name'],ind_rets=ret_retailers['Retailer_Name'])


#Function called in get price

def GetPriceWalmart(main_prod):
    ret="Walmart"
    ret_prods = get_ret_prod_catalog()
    ret_retailers = get_ret_retailer()

    # main_prod = "Royal Purple Max-EZ PSF"
    # ret="Walmart"

    prod=re.sub('[^a-zA-Z0-9 \n\.]', ' ', main_prod)
    
    URL = ret_retailers.loc[ret_retailers['Retailer_Name']==ret,['URL']].values[0]
    domain= URL[0]
    URL = domain + "/search?q=" + prod.replace(" ","+")
    
    base_img_url = ret_prods.loc[ret_prods['Product_Name']==main_prod,['base_img_url']].values[0]
    
    # b black f5 mr1 mr2-xl lh-copy f4-l

    desc_id = "mb1 ph1 pa0-xl bb b--near-white w-25" #ret_data.loc[ret_data['Retailer_Name']==ret,['desc_id']].values[0]
    
    # Read the search result page
    req = Request(URL , headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
    webpage = urlopen(req).read()
    page_soup = BeautifulSoup(webpage, "html.parser")
    page_urls = page_soup.findAll(name='a')
    page_urls = [img.get('href') for img in page_urls if img.get('href').startswith("/ip/")]

    print(page_urls)
    img_dict={}
    for p in page_urls:
        temp_ = []
        p = domain + p
        try:
            req = Request(p , headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
            webpage = urlopen(req).read()
            page_soup = BeautifulSoup(webpage, "html.parser")
            # img_urls = page_soup.findAll(name='img')
            # img_urls = [img.get('src') for img in img_urls]
            # for a in page_soup.find_all("a" , href=True):
            #     if re.findall(r".+(?=jpg|png|jpeg)",a['href']): 
            #         img_urls.append(a['href'])
            # img_urls = list(set(img_urls))
            img_div = page_soup.find("div",class_="ma2")   #mr3 ml7 self-center
            img_urls = img_div.findAll(name="img")
            img_urls = [img.get('src') for img in img_urls]
            temp_ = [[img,compare_images(base_img_url[0],img)] for img in img_urls]
            #temp_ = [img,compare_images(base_img_url[0],img)] 
            temp_ = max(temp_,key=lambda x: x[1])
            img_dict[p] = temp_           
        except:
            pass

    max_score=0.0
    max_img=""
    max_key=""
    final_img_dict={}
    for k,v in img_dict.items():
        if v[1]>max_score:
            max_img=v[0]
            max_score=v[1]
            max_key=k

    final_img_dict[max_key] = max_img
    URL = list(final_img_dict.keys())[0]
    print(URL)
    req = Request(URL , headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
    webpage = urlopen(req).read()
    page_soup = BeautifulSoup(webpage, "html.parser")
    
    desc = page_soup.find("div",class_="mr3 ml7 self-center")
    
    pr = str(page_soup.find("span",itemprop="price").text)

    
    return base_img_url, final_img_dict, pr, max_score, max_key


def get_distributors():
    dists=[]
    query = "GEAR SAVER HYPOID G/O 85W-140 12/1 LT" + "+Bel-Ray"

    for j in search(query, num_results=20):
	    dists.append(j)

    return dists


    dists = get_distributors()
    df = pd.DataFrame(columns=['Source','Target','Type','Weight'])
    df['Target']= dists
    df['Type']="undirected"
    df['Weight']=1
    df['Source']= prod + "/n" + brand


def get_img():
    path = os.path.join(basedir, 'templates/add1.html')
    with open(path, 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
    
    #ps = re.findall("[A-Z]{1,2}[0-9][A-Z0-9]? [0-9][ABD-HJLNP-UW-Z]{2}", text)
    ps = soup.findAll(name='img')
    print(ps)

@application.route('/callback',methods=['POST','GET'])
@login_required
def cb():
    return gm(request.args.get('data'))

@application.route('/callback3',methods=['POST','GET'])
@login_required
def db1():
    prod=request.args.get('prod')
    brand=request.args.get('brand')
    date=request.args.get('date')
    return geo_plot(prod=prod,brand=brand,date=date)


@application.route('/callback2',methods=['POST','GET'])
@login_required
def country():
    prod = request.args.get('prod')
    brand = request.args.get('brand')
    return geo_plot(prod=prod,brand=brand)


@application.route('/callback4',methods=['POST','GET'])
@login_required
def rb():
    prod=request.args.get('prod')
    brand=request.args.get('brand')
    return prophet_plot(prod=prod,brand=brand)

def get_country(domain):
    url="https://check-host.net/ip-info?host=" + domain
    req = Request(url , headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
    webpage = urlopen(req).read()
    page_soup = BeautifulSoup(webpage, "html.parser")
    page_soup = page_soup.find('div',id="ip_info-dbip")
    tds = page_soup.find('td', class_="inside_info_ipinfo").find_all('td')
    country=""
    for td in tds:
        if "Country" in td:
            country=td.find_next_sibling('td').text.strip()
    time.sleep(10)
    print("get")
    return country


def geo_plot(prod="GEAR SAVER TRANS OIL 80W 12/1 LT",brand="Bel-Ray",date=date.today()):

    date = pd.to_datetime("25/11/2021").date() # To be removed later
    date = pd.to_datetime(date) if type(date)!=pd.datetime else date
    df = pd.read_csv(os.path.join(basedir + '/data/ind_dist_data.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date

    dists = df.loc[(df['Brand']==brand) & (df['Date']==date) & (df['Product']==prod),'Countries'].values[0]
    dists = dists.split("_")
    counts = dict()
    for i in dists:
        counts[i] = counts.get(i, 0) + 1
    df = pd.DataFrame(columns=(['Country','No']))
    df['Country'] = counts.keys()
    df['No'] = counts.values()
    fig = px.pie(df, values='No', names='Country', title='Distributors by countries')
    graphJSON2 = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
   
    return graphJSON2


def gm(prod="GEAR SAVER TRANS OIL 80W 12/1 LT"):
    brnd="Bel-Ray"
    df = pd.read_csv((os.path.join(application.config['DATA_PATH'], 'ind_dist_data.csv')))
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    df = df.sort_values(by="Date",ascending=False)
    d1= df[(df['Brand']==brnd) & (df['Product']==prod)]
    d2= df[(df['Brand']=="Motul") & (df['Product']==prod)]
    fig = px.line(df[(df['Product']==prod)],x="Date",y="Number of Distributors", title=prod,color='Brand')
    #fig = px.line(df[(df['Brand']==brnd) & (df['Product']==prod)],x="Date",y="Number of Distributors", title=prod,color='Brand)

    #fig = px.line(d1[(d1['Brand']==brnd) & (d1['Product']==prod)],x="Date",y="Number of Distributors", title=prod)
    #fig.add_scatter(x=d2['Date'], y=d2['Number of Distributors'], mode='lines',showlegend=True)
    # fig.update_xaxes(
    # title=prod)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@application.route('/dashboard')
@login_required
def dashboard():    
    prods = get_prod_catalog()
    ind_rets = get_brand_data()
    base = pd.to_datetime("25/11/2021").date() # To be removed later date.today()
    date_list = [base-timedelta(days=1) - timedelta(days=x) for x in range(50)]
    best_prod,avg_dists,best_cntr,best_val = key_ind_stats()
    return render_template('ind_dashboard_home_template.html',graphJSON=gm(),graphJSON2=geo_plot(),graphJSON3=prophet_plot(),prods=prods['Product_Name'],ind_rets=ind_rets.loc[ind_rets['Calumet_Brand']=="Y",'Brand_Name'],ind_comp_rets=ind_rets.loc[ind_rets['Calumet_Brand']=="N",'Brand_Name'],dates=date_list,tdate=base,best_prod=best_prod,avg_dists=avg_dists,best_cntr=best_cntr.upper(),best_val=best_val)

@application.route('/ret_dashboard')
@login_required
def ret_dashboard():    
    ret_prods = get_ret_prod_catalog()
    ret_retailers = get_ret_retailer()
    base = pd.to_datetime("25/11/2021").date() # To be removed later date.today()
    date_list = [base-timedelta(days=1) - timedelta(days=x) for x in range(50)]
    best_prod,avg_dists,best_cntr,best_val = key_ind_stats()
    return render_template('ret_dashboard_home_template.html',graphJSON=gm(),graphJSON2=geo_plot(),graphJSON3=prophet_plot(),prods=ret_prods['Product_Name'],ind_rets=ret_retailers['Retailer_Name'],dates=date_list,tdate=base,best_prod=best_prod,avg_dists=avg_dists,best_cntr=best_cntr.upper(),best_val=best_val)


def key_ind_stats():
    start_date = pd.to_datetime("25/11/2021").date() # To be removed later date.today()
    end_date = start_date-timedelta(days=7)
    df = pd.read_csv(os.path.join(basedir + '/data/ind_dist_data.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    mask = (df['Date'] < start_date) & (df['Date'] > end_date)
    subset = df.loc[(df['Brand']=="Bel-Ray") & (mask),['Product','Number of Distributors']]
    subset = subset.groupby(['Product'], as_index=False).mean().sort_values(by="Number of Distributors",ascending=False)
    subset = subset.max()

    csubset = df.loc[(df['Brand']=="Bel-Ray") & (mask),['Product','Countries']]

    cdict={}
    csubset['MaxC'] = 0
    csubset['MaxCntr'] = ""
    csubset['MaxVal'] = ""
    csubset['MaxC'] = csubset['Countries'].apply(cal_country_dict)
    csubset['MaxCntr'] = csubset['MaxC'].apply(lambda x: x.split("_")[0])
    csubset['MaxVal'] = csubset['MaxC'].apply(lambda x: x.split("_")[1])
    csubset = csubset.sort_values(['MaxC'],ascending=False)
    
    return subset['Product'], subset['Number of Distributors'],csubset.iloc[0]['MaxCntr'], csubset.iloc[0]['MaxVal']

def cal_country_dict(row):
    try:
        dists = row.split("_")

        counts = dict()
        for i in dists:
            counts[i]=counts.get(i,0) +1
        
        maxc = max(counts)
        maxval = maxc+ "_"+str(counts[maxc])
        # print(str(counts[maxc]))
        # print(maxc)
        # print(maxc+"_"+str(counts[maxc]))
    except:
        maxval=" _"+"0"
    return maxval
    
def prophet_plot(prod="GEAR SAVER TRANS OIL 80W 12/1 LT",brand="Bel-Ray"):
    
    df = pd.read_csv(os.path.join(basedir + '/data/ind_dist_data.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date

    #dists = df.loc[(df['Brand']=="Bel-Ray") &  (df['Product']=="GEAR SAVER TRANS OIL 80W 12/1 LT")]

    dists = df.loc[(df['Brand']==brand) &  (df['Product']==prod)]

    df = pd.DataFrame(columns=(['ds','y']))
    df['ds'] = dists['Date']
    df['y'] = dists['Number of Distributors']

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=365)

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    # Create the plotly figure
    yhat = go.Scatter(
    x = forecast['ds'],
    y = forecast['yhat'],
    mode = 'lines',
    marker = {
        'color': '#3bbed7'
    },
    line = {
        'width': 3
    },
    name = 'Forecast',
    )

    yhat_lower = go.Scatter(
    x = forecast['ds'],
    y = forecast['yhat_lower'],
    marker = {
        'color': 'rgba(0,0,0,0)'
    },
    showlegend = False,
    hoverinfo = 'none',
    )

    yhat_upper = go.Scatter(
    x = forecast['ds'],
    y = forecast['yhat_upper'],
    fill='tonexty',
    fillcolor = 'rgba(231, 234, 241,.75)',
    name = 'Confidence',
    hoverinfo = 'none',
    mode = 'none'
    )

    actual = go.Scatter(
    x = df['ds'],
    y = df['y'],
    mode = 'markers',
    marker = {
        'color': '#fffaef',
        'size': 4,
        'line': {
        'color': '#000000',
        'width': .75
        }
    },
    name = 'Actual'
    )

    layout = go.Layout(
    yaxis = {
        'title': "Number of Distributors"
        # 'tickformat': format(y),
        # 'hoverformat': format(y)
    },
    hovermode = 'x',
    # xaxis = {
    #     'title': agg.title()
    # },
    margin = {
        't': 20,
        'b': 50,
        'l': 60,
        'r': 10
    },
    legend = {
        'bgcolor': 'rgba(0,0,0,0)'
    }
    )
    
    data = [yhat_lower, yhat_upper, yhat, actual]

    fig = dict(data = data, layout = layout)

    graphJSON3 = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
   
    return graphJSON3

@application.route('/scrape',methods=['POST','GET'])
@login_required
def start_scrapper():
    prod_link_path = os.path.join(basedir, 'scrapper/retail_product_link_data.csv')
    retailers_path = os.path.join(basedir, 'scrapper/retailers.csv')
    aggregated_data_path= os.path.join(basedir,'scrapper/aggregated_price_data.csv')
    log_path = os.path.join(basedir,'scrapper/main_log.csv')
    keywords = {"trans":"transmission","perf":"performance","min":"mineral","eng":"engine","syn":"synthetic","rac":"racing","est":"ester","Ounce": "oz"}

    #scrapper = Scrapper(prod_link_path,retailers_path,aggregated_data_path,log_path,keywords)
    data = Process_Data(prod_link_path, keywords)

    #data = data.loc[data['competitor_product']=="Red Line WaterWetter Pink Concentrate Coolant/Antifreeze"]

    data = data.iloc[1:4,:]

    retailers_data = pd.read_csv(retailers_path,index_col=None)
    
    for retailer in retailers_data['retailer_name']:
        start_time=time.time()
        retailer=retailer
        retailer_URL = retailers_data.loc[retailers_data['retailer_name']==retailer]['URL'].values[0]
        desc_id = retailers_data.loc[retailers_data['retailer_name']==retailer]['desc_id'].values[0]
        temp_data=data.copy()
        temp_data['pricecomp'] = temp_data.apply(GetPrice2,axis=1)
        temp_data['pricecalu'] = temp_data.apply(GetPrice3,axis=1)
        temp_data['datetime'] = pd.to_datetime('today').strftime("%m/%d/%Y, %H:%M:%S")

        print('temp data here: ')
        print(temp_data)

        agg_cols = ['calumet_product','competitor_product','pricecomp','pricecalu','datetime']

        fail_count = len(temp_data.loc[temp_data['pricecomp']==0])
        fail_count = len(temp_data.loc[temp_data['pricecalu']==0])
        pass_count = len(temp_data)-fail_count

        # read the aggregated file
        if os.path.exists(aggregated_data_path):
            agg_data = pd.read_csv(aggregated_data_path,index_col=None)
        else:
            agg_data = pd.DataFrame(columns=agg_cols)
        

        agg_data = agg_data.append(temp_data,ignore_index=True)
        agg_data['retailer']=retailer
        print(agg_data)

        agg_data.to_csv(aggregated_data_path,index=False)
        
        run_time = time.time() - start_time
        status="Fetched data for all products" if fail_count==0 else "Passed for {} products, Failed for {} products".format(pass_count,fail_count)
        ScrapperLogs(log_path,retailer, pd.to_datetime('today').strftime("%m/%d/%Y, %H:%M:%S"),run_time,status)

        return redirect(url_for('scrapper_log'))

    #return render_template('scrapper_log.html',tables=[agg_data.to_html(table_id="dist_table")], titles=agg_data.columns.values)



@application.route('/slog')
@login_required
def scrapper_log():
    mlog = pd.read_csv(os.path.join(application.config['SCRAPPER_PATH'], 'main_log.csv'),index_col=False)
    mlog = mlog.sort_values(by="Date",ascending=False)
    mlog = mlog.set_index("Date")
    return render_template('scrapper_log.html',tables=[mlog.to_html(table_id="dist_table")], titles=mlog.columns.values)



if __name__ == "__main__":
    #application.run()
    socketio.run(application,debug=True, port=5007,host="localhost")



# def gm(country="United Kingdom"):
#     df = pd.DataFrame(px.data.gapminder())
#     fig = px.line(df[df['country']==country],x="year",y="gdpPercap")
#     graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#     return graphJSON