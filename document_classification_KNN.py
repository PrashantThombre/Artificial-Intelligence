import urllib2
from bs4 import BeautifulSoup
import re
import string
import math

def getJobList():

    # SCRAPE Job URLs from ACM.ORG
    joburls = list()
    acm_jobs_page = 'http://jobs.acm.org/jobs/results/keyword/artificial+intelligence+engineer?location=California'
    acm_page = urllib2.urlopen(acm_jobs_page)
    soup = BeautifulSoup(acm_page, 'html.parser')
    for div in soup.find_all(name='div', attrs={'class': 'aiClearfix aiDevFeaturedSection '}):
        for div2 in div.find_all(name='div', attrs={'class': 'aiResultTitle aiDevIconSection '}):
            for a in div2.find_all('a'):
                #print (a.text)
                joburls.append(a["href"])
                #print ("")
    all_documents = list()
    
    # From these Job URLs, extract the corresponding JDs
    for joburl in joburls:
        acm_job_description_page = 'http://jobs.acm.org'+joburl;
        print "--" * 40
        print joburl
        print "--" * 40
        acm_description_page = urllib2.urlopen(acm_job_description_page)
        soup = BeautifulSoup(acm_description_page,'html.parser')
        for div in soup.find_all(name='div',attrs={'class':'aiListingTabContainer','id':'detailTab'}):
            for description in div.find_all(name='div',attrs={'id':'detailTab'}):
                jdText = description.text
                jdText = re.sub('[^A-Za-z]+', ' ', jdText)
                jdText = jdText.replace('\n', ' ')
                all_documents.append(jdText)
    print joburls
    print all_documents

getJobList()
