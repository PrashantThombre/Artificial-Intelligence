import urllib2
from bs4 import BeautifulSoup
import re
import math
import pickle


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
        # print all_documents
        # print all_urls
        # print all_titles
        return all_documents,all_urls, all_titles

    # Pickle enables us to write Python objects directly to a file: Here predefined_table is a dictionary of tuples of timestamp, Titles, URLs, and JDs
    def build_table(self,predefined_table):
        fileObj = open('datafile.pkl','ab')
        pickle.dump(predefined_table,fileObj)
        fileObj.close()

    # Function to check if the keyword given by the user is already present in the table; if yes return the corresponding job details by unpickling
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

        # Mathematical formula for cosine similarity
        def cosine_similarity(vector1, vector2):
            dot_product = sum(p * q for p, q in zip(vector1, vector2))
            magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(sum([val ** 2 for val in vector2]))
            if not magnitude:
                return 0
            return dot_product / magnitude

        # KNN starts from here actually
        tfidf_representation = tfidf(all_documents)
        key_tfidf_representation = tfidf(key_document)
        our_tfidf_comparisons = []

        # Compute the cosine_distance = 1 - cosine_similarity  between the KEY_DOC and all the other documents
        for count_1, doc_1 in enumerate(key_tfidf_representation):
            for count_0, doc_0 in enumerate(tfidf_representation):
                our_tfidf_comparisons.append((1-cosine_similarity(doc_0, doc_1), count_0, count_1))

        # Sort the distances in ascending order and pick the k documents ordered on lowest distances
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
