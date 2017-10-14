import argparse as ap

import urllib2
import time
from bs4 import BeautifulSoup
import re

import math
import os
import pickle

from collections import defaultdict
from math import sqrt
import random

class Agent:
    def scrape_jobs(self,keywords):
        print keywords
        acm_job_url = []
        acm_job_title = []
        acm_job_descriptions = []
        ieee_job_url = []
        ieee_job_title = []
        ieee_job_descriptions = []
        indeed_job_url = []
        indeed_job_title = []
        indeed_job_descriptions = []

        split_words = keywords.split()
        try:
            # Here we are scraping the ACM website for each sequence of keywords in [keywords_list]
            acm_url = 'http://jobs.acm.org/jobs/results/keyword/' + '+'.join(split_words) + '/California?radius=0'
            acm_page = urllib2.urlopen(acm_url)
            acm_soup = BeautifulSoup(acm_page, 'html.parser')


            for div in acm_soup.find_all(name='div', attrs={'class': 'aiClearfix aiDevFeaturedSection '}):
                for div2 in div.find_all(name='div', attrs={'class': 'aiResultTitle aiDevIconSection '}):
                    for a in div2.find_all('a'):
                        acm_job_url.append('http://jobs.acm.org'+a["href"])
                        acm_job_title.append(a.text)

            for url in acm_job_url:
                acm_job_description_page = url
                acm_description_page = urllib2.urlopen(acm_job_description_page)
                soup = BeautifulSoup(acm_description_page, 'html.parser')
                for div in soup.find_all(name='div', attrs={'class': 'aiListingTabContainer', 'id': 'detailTab'}):
                    for description in div.find_all(name='div', attrs={'id': 'detailTab'}):
                        jobText = description.text
                        jobText = re.sub('[^A-Za-z]+', ' ', jobText)
                        jobText = jobText.replace('\n', ' ')
                        acm_job_descriptions.append(jobText)
        except AttributeError:
            print "ACM job scraping error: Valid element not found on the job page"
        except Exception:
            print "ACM job scraping error: No job found"

        try:
            # Here we are scraping the IEEE website for each sequence of keywords in [keywords_list]
            ieee_url = 'http://jobs.ieee.org/jobs/results/keyword/' + '+'.join(split_words) + '/California?radius=0&EquivState=CA&view=List_Brief'
            ieee_page = urllib2.urlopen(ieee_url)
            ieee_soup = BeautifulSoup(ieee_page, 'html.parser')


            table = ieee_soup.find(name = 'table', attrs={'id':'aiResultsBrief'} )
            for row in table.find_all(name='tr', attrs={'class':'aiResultsRow '}):
                #Featured Jobs are listed under different class
                for td in row.find_all(name='td', attrs={'class': 'aiResultsJobTitle aiDevFeaturedSection aiDevIconSection aiFeaturedBriefView '}):
                    for a in td.find_all('a'):
                        ieee_job_url.append('http://jobs.ieee.org' + a["href"])
                        ieee_job_title.append(a.text)
                #Rest of the jobs are given under this class
                for td in row.find_all(name='td', attrs={'class': 'aiResultsJobTitle aiDevFeaturedSection aiDevIconSection '}):
                    for a in td.find_all('a'):
                        ieee_job_url.append('http://jobs.ieee.org' + a["href"])
                        ieee_job_title.append(a.text)

            for url in ieee_job_url:
                ieee_job_description_page = url
                ieee_description_page = urllib2.urlopen(ieee_job_description_page)
                soup = BeautifulSoup(ieee_description_page, 'html.parser')
                for div in soup.find_all(name='div', attrs={'class': 'aiListingTabContainer', 'id': 'detailTab'}):
                    for description in div.find_all(name='div', attrs={'id': 'detailTab'}):
                        jobText = description.text
                        jobText = re.sub('[^A-Za-z]+', ' ', jobText)
                        jobText = jobText.replace('\n', ' ')
                        ieee_job_descriptions.append(jobText)
        except AttributeError:
            print "IEEE job scraping error: Valid element not found on the job page"
        except Exception:
            print "IEEE job scraping error: No job found"

        try:
            #Using Mobile website URL for consistency between the URLs and to easily get job description
            indeed_url = 'https://www.indeed.com/m/jobs?q=' + '+'.join(split_words) + '&l=California'
            indeed_page = urllib2.urlopen(indeed_url)
            indeed_soup = BeautifulSoup(indeed_page, 'html.parser')

            # Here we are scraping the Indeed website for each sequence of keywords in [keywords_list]
            for head in indeed_soup.find_all(name='h2', attrs={'class': 'jobTitle'}):
                for a in head.find_all('a'):
                    indeed_job_url.append('https://www.indeed.com/m/'+a["href"])
                    indeed_job_title.append(a.text)

            for url in indeed_job_url:
                indeed_job_description_page = url
                indeed_description_page = urllib2.urlopen(indeed_job_description_page)
                soup = BeautifulSoup(indeed_description_page, 'html.parser')
                for div in soup.find_all(name= 'div', attrs={'id':'desc'}):
                    jobText = div.text
                    indeed_job_descriptions.append(jobText)
        except AttributeError:
            print "Indeed job scraping error: Valid element not found on the job page"
        except Exception:
            print "Indeed job scraping error: No job found"

        all_documents = acm_job_descriptions + ieee_job_descriptions + indeed_job_descriptions
        all_urls = acm_job_url + ieee_job_url + indeed_job_url
        all_titles = acm_job_title + ieee_job_title + indeed_job_title
        #print all_documents
        #print all_urls
        #print all_titles
        return all_documents,all_urls, all_titles

    #Pickle enables us to write Python objects directly to a file: Here predefined_table is a dictionary of tuples of timestamp, Titles, URLs, and JDs
    def build_table(self,predefined_table):
        fileObj = open('cs256datafile.pkl','ab')
        pickle.dump(predefined_table,fileObj)
        fileObj.close()

    #Function to check if the keyword given by the user is already present in the table; if yes return the corresponding job details by unpickling
    def isKeywordPresent(self,cmd_keywords):
            loc_all_titles = list()
            loc_all_urls = list()
            loc_all_documents = list()
            table_all = list()
            try:
                infile = open('cs256datafile.pkl', 'rb')
                while 1:
                    try:
                        table_all.append(pickle.load(infile))
                    except (EOFError):
                        break
                infile.close()

                #ret_table = pickle.load(open('cs256datafile.pkl','rb'))
                for ret_table in table_all:
                    if cmd_keywords in ret_table:
                        unzipped_list = zip(*ret_table[cmd_keywords])
                        loc_all_titles = list(unzipped_list[1])
                        loc_all_urls = list(unzipped_list[2])
                        loc_all_documents = list(unzipped_list[3])
                        return loc_all_documents, loc_all_urls, loc_all_titles
                #The code to verify the timestamp should go here. Due to lack of time I have not implemented it in this version of the code
                return loc_all_documents, loc_all_urls, loc_all_titles
            except Exception:
                print "Unable to read the table"
                pass

    def performKNN(self, doc_List, url_List, title_List, cmd_keywords, cmd_k=15):
        #Tokenize assigns each word in doc a unique number
        tokenize = lambda doc: doc.lower().split(" ")

        #A Very Short Document formed of the Keywords
        key_document = cmd_keywords
        key_document = [key_document]
        all_documents = doc_List
        all_urls = url_List
        all_titles = title_List

        #Counts the number of times a word has occurred in the tokenized document using the count function on list
        def term_frequency(term, tokenized_document):
            return tokenized_document.count(term)

        #Converts the plain word count into a linear frequency of terms. [Optional] We could have also worked with only the term frequency as well
        def sublinear_term_frequency(term, tokenized_document):
            count = tokenized_document.count(term)
            if count == 0:
                return 0
            return 1 + math.log(count)

        #Inverse document frequency reduced the natural bias of a larger document by taking into consideration all the words in all our documents
        def inverse_document_frequencies(tokenized_documents):
            idf_values = {}
            all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
            for tkn in all_tokens_set:
                contains_token = map(lambda doc: tkn in doc, tokenized_documents)
                idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))
            return idf_values

        #Main Function for TFIDF computation: First tokenizes the documents, then computes the IDF and TF, then computes TFIDF as the dot product of TF and IDF
        def tfidf(documents):
            tokenized_documents = [tokenize(d) for d in documents]
            idf = inverse_document_frequencies(tokenized_documents)
            tfidf_documents = []
            for document in tokenized_documents:
                doc_tfidf = []
                for term in idf.keys():
                    tf = sublinear_term_frequency(term, document)
                    doc_tfidf.append(tf * idf[term])
                tfidf_documents.append(doc_tfidf)
            return tfidf_documents

        #Mathematical formula for cosine similarity
        def cosine_similarity(vector1, vector2):
            dot_product = sum(p * q for p, q in zip(vector1, vector2))
            magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(sum([val ** 2 for val in vector2]))
            if not magnitude:
                return 0
            return dot_product / magnitude

        #KNN starts from here actually
        tfidf_representation = tfidf(all_documents)
        key_tfidf_representation = tfidf(key_document)
        our_tfidf_comparisons = []

        #Compute the cosine_distance = 1 - cosine_similarity  between the KEY_DOC and all the other documents
        for count_1, doc_1 in enumerate(key_tfidf_representation):
            for count_0, doc_0 in enumerate(tfidf_representation):
                our_tfidf_comparisons.append((1-cosine_similarity(doc_0, doc_1), count_0, count_1))

        #Sort the distances in ascending order and pick the k documents ordered on lowest distances
        try:
            k = int(cmd_k)
            ret_documents = list()
            ret_urls = list()
            ret_titles = list()
            our_tfidf_comparisons = sorted(our_tfidf_comparisons)
            for i in xrange(0, k):
                ret_documents.append(all_documents[our_tfidf_comparisons[i][1]])
                ret_urls.append(all_urls[our_tfidf_comparisons[i][1]])
                ret_titles.append(all_titles[our_tfidf_comparisons[i][1]])
            return ret_documents, ret_urls, ret_titles
        except IndexError:
            print "Number of jobs retrieved from Indeed, ACM, and IEEE are less than the K value (15 for clustering)"
            print "Please, try a new keyword"


class Environment:
    all_documents = list()
    all_urls = list()
    all_titles = list()

    def __init__(self):
        self.agentObj = Agent()

    #Convert a vector with may zeros to a dense vector
    def densify(self, x, n):
        d = [0] * n
        for i, v in x:
            d[i] = v
        return d

    #c is the cluster center, x is the vector to be clustered
    #compute the euclidean distance, sqrt() is optional, can be avoided for efficiency
    def square_distance(self, x, c):
        eucdist = 0.
        for i, v in x:
            eucdist += (v - c[i]) ** 2
        return sqrt(eucdist)

    def compute_mean(self, xs, l):
        c = [0.] * l
        n = 0
        for x in xs:
            for i, v in x:
                c[i] += v
            n += 1
        for i in xrange(l):
            c[i] /= n
        return c

    def kmeans(self, k, xs, l, n_iter=10):
        # Initialize from random centers.
        center_list = [self.densify(xs[i], l) for i in random.sample(xrange(len(xs)), k)]
        group = [None] * len(xs)
        # Number of iterations is 10 by default
        for _ in xrange(n_iter):
            for i, x in enumerate(xs):
                group[i] = min(xrange(k), key=lambda j: self.square_distance(xs[i], center_list[j]))
            for j, c in enumerate(center_list):
                mem_group = (x for i, x in enumerate(xs) if group[i] == j)
                center_list[j] = self.compute_mean(mem_group, l)
        return group

    #Code to write HTML page for Jobs returned by KNN
    def write_html(self,title_list,url_list):
        print "Writing To The HTML File.."
        yield '<table border = 1>'
        yield '<h3> Top Jobs: </h3> <br>'
        for title,url in zip(title_list,url_list):
            yield '<tr><td>'
            yield title
            yield '</td><td><a href='+url+'>'
            yield url
            yield  '</a></td></tr>'
        yield '</table>'

    #Code to write HTML page for Jobs returned After Clustering
    def write_html_cluster(self,title_list,url_list,cluster_id):
        print "Writing Clusters To The HTML File.."
        yield '<table border = 1>'
        yield '<h3> Cluster: '
        yield cluster_id
        yield '</h3> <br>'
        for title,url in zip(title_list,url_list):
            yield '<tr><td>'
            yield title
            yield '</td><td><a href='+url+'>'
            yield url
            yield  '</a></td></tr>'
        yield '</table>'

    def main(self,cmd_keywords,cmd_k):
        try:
            predefined_table = dict()
            #Checking if the pkl file is already present in the current directory
            if os.path.isfile('cs256datafile.pkl'):
                #If yes then check for the keyword in that file
                self.all_documents, self.all_urls, self.all_titles = self.agentObj.isKeywordPresent(cmd_keywords)
                #if keyword not found scrape the web and call build_table on it to add this entry to the table
                if not self.all_documents:
                    self.all_documents, self.all_urls, self.all_titles = self.agentObj.scrape_jobs(cmd_keywords)
                    #Put timestamp, URL, Title, and JD together in a dictionary keyed on the Keyword sequence
                    consolidated_doc = zip([time.time()] * len(self.all_urls), self.all_titles, self.all_urls, self.all_documents)
                    predefined_table[cmd_keywords] = consolidated_doc
                    self.agentObj.build_table(predefined_table)
            else:
                #Build the table if the file is not present already
                print "Building the table based on predefined keywords..."
                #Only 2 predefined keywords given to reduce the time required to build the table, appending the commandline keyword while building the first table
                predefined_keywords = ['data science', 'machine learning engineer']
                predefined_keywords.append(cmd_keywords)
                for keywords in predefined_keywords:
                    documents, urls, titles = self.agentObj.scrape_jobs(keywords)
                    if keywords == cmd_keywords:
                        self.all_documents, self.all_urls, self.all_titles =documents, urls, titles
                    consolidated_doc = zip([time.time()] * len(urls), titles, urls, documents)
                    predefined_table[keywords] = consolidated_doc
                print predefined_table
                self.agentObj.build_table(predefined_table)

            rec_docs, rec_urls, rec_titles = self.agentObj.performKNN(self.all_documents, self.all_urls, self.all_titles, cmd_keywords, cmd_k)
            print 'The Top ' + str(cmd_k) + ' jobs are:\n'
            print rec_urls

            try:
                html_code = '\n'.join(self.write_html(rec_titles,rec_urls))
                html_file = open("cs265_hw2_knn.html", "w")
                html_file.write(html_code)
                html_file.close()
                print "HTML File Written: cs265_hw2_knn.html"
            except Exception:
                print "Writing to HTML file failed in KNN mode."
        except Exception:
            print "Unexpected Error occurred in program"

        try:
            print "\nClustering Mode"
            rec_docs, rec_urls, rec_titles = self.agentObj.performKNN(self.all_documents, self.all_urls,self.all_titles, cmd_keywords)
            vocab = {}
            xs = []
            clustering_documents = rec_docs
            clustering_urls = rec_urls
            printURLs = list()
            clustering_titles = rec_titles
            printTitles = list()
            #Fixed the number of clusters to 3 as in the specs
            k = 3
            for doc_sample in clustering_documents:
                x = defaultdict(float)
                for w in re.findall(r"\w+", doc_sample):
                    vocab.setdefault(w, len(vocab))
                    x[vocab[w]] += 1
                xs.append(x.items())
            cluster_ind = self.kmeans(k, xs, len(vocab))
            clusters = [set() for _ in xrange(k)]
            for i, j in enumerate(cluster_ind):
                clusters[j].add(i)

            html_file = open("cs265_hw2_clusters.html", "w")
            for j, c in enumerate(clusters):
                print("cluster %d:" % j)
                for i in c:
                    print("\t%s" % clustering_documents[i])
                    printURLs.append(clustering_urls[i])
                    printTitles.append(clustering_titles[i])
                try:
                    html_code = '\n'.join(self.write_html_cluster(printTitles, printURLs, str(j)))

                    html_file.write(html_code)
                    print "HTML File Written: cs265_hw2_clusters.html"
                except Exception:
                    print "Writing to HTML file failed in clustering mode."
            html_file.close()
        except Exception:
            print "Error while clustering"

if __name__ == "__main__":
    obj = Environment()
    #Required arguments k and w for K-Value and keyword sequence
    parser = ap.ArgumentParser(description='Career and Job Agent')
    parser.add_argument('-k', '--k_Val', help='Value of k, #top_jobs_required', required=True)
    parser.add_argument('-w', '--keywords', help='List of keywords', nargs='+', required=True)
    args = parser.parse_args()

    cmd_keywords = args.keywords
    cmd_keywords = " ".join(cmd_keywords)
    cmd_keywords = cmd_keywords.lower()
    print cmd_keywords

    cmd_k = int(args.k_Val)
    #While loop to persistently run the application until user enters a quit signal. In our case it is any key other that 1
    while 1:
        obj.main(cmd_keywords,cmd_k)
        choice = raw_input('Select a choice: \n[Press 1] - New Keyword Search \n[Press Any Other Key] - Quit The Application\n')
        if choice == '1':
            cmd_keywords = raw_input('Enter the new keywords\n').lower()
            cmd_k = input('Enter the value of k\n')
        else:
            quit()
    #Multi threading failed to scrape the jobs in parallel. I had to wait until the scraping finished.
    # scraping_thread = threading.Thread(target=obj.scrape_jobs,args=(predefined_keywords,))
    # scraping_thread.start()
    # while(scraping_thread.isAlive()):
    #     print "a."
    #print all_documents
    #print all_urls


#REFERENCES:
# https://stackoverflow.com/questions/25674169/how-does-the-list-comprehension-to-flatten-a-python-list-work
# https://gist.github.com/larsmans/4952848 --> Reference for K Means Clustering
# http://billchambers.me/tutorials/2014/12/22/cosine-similarity-explained-in-python.html --> Cosine similarity computation, Bill Chambers, 22 December 2014
