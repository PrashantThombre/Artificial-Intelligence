from agent import Agent

import time
import re
import os

from collections import defaultdict
from math import sqrt
import random


class Environment:
    all_documents = list()
    all_urls = list()
    all_titles = list()

    def __init__(self):
        self.agentObj = Agent()

    # Convert a vector with may zeros to a dense vector
    def densify(self, x, n):
        d = [0] * n
        for i, v in x:
            d[i] = v
        return d

    # c is the cluster center, x is the vector to be clustered
    # compute the euclidean distance, sqrt() is optional, can be avoided for efficiency
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
                mem_group = (x for t, x in enumerate(xs) if group[t] == j)
                center_list[j] = self.compute_mean(mem_group, l)

        return group

    # Code to write HTML page for Jobs returned by KNN
    def write_html(self,title_list,url_list):
        print "Writing To The HTML File.."
        yield '<table border = 1>'
        yield '<h3> Top Jobs: </h3> <br>'
        for title, url in zip(title_list,url_list):
            yield '<tr><td>'
            yield title
            yield '</td><td><a href=' + url + '>'
            yield url
            yield  '</a></td></tr>'
        yield '</table>'

    # Code to write HTML page for Jobs returned After Clustering
    def write_html_cluster(self, title_list, url_list, cluster_id):
        print "Writing Clusters To The HTML File.."
        yield '<table border = 1>'
        yield '<h3> Cluster: '
        yield cluster_id
        yield '</h3> <br>'
        for title, url in zip(title_list, url_list):
            yield '<tr><td>'
            yield title
            yield '</td><td><a href='+url+'>'
            yield url
            yield '</a></td></tr>'
        yield '</table>'

    def main(self, cmd_keywords, cmd_k):
        try:
            predefined_table = dict()
            # Checking if the pkl file is already present in the current directory
            if os.path.isfile('./resource/datafile.pkl'):
                # If yes then check for the keyword in that file
                self.all_documents, self.all_urls, self.all_titles = self.agentObj.isKeywordPresent(cmd_keywords)
                # if keyword not found scrape the web and call build_table on it to add this entry to the table
                if not self.all_documents:
                    self.all_documents, self.all_urls, self.all_titles = self.agentObj.scrape_jobs(cmd_keywords)
                    # Put timestamp, URL, Title, and JD together in a dictionary keyed on the Keyword sequence
                    consolidated_doc = zip([time.time()] * len(self.all_urls), self.all_titles, self.all_urls, self.all_documents)
                    predefined_table[cmd_keywords] = consolidated_doc
                    self.agentObj.build_table(predefined_table)
            else:
                # Build the table if the file is not present already
                print "Building the table based on predefined keywords..."
                # Only 2 predefined keywords given to reduce the time required to build the table, appending the commandline keyword while building the first table
                predefined_keywords = ['data science', 'machine learning engineer']
                predefined_keywords.append(cmd_keywords)
                for keywords in predefined_keywords:
                    documents, urls, titles = self.agentObj.scrape_jobs(keywords)
                    if keywords == cmd_keywords:
                        self.all_documents, self.all_urls, self.all_titles = documents, urls, titles
                    consolidated_doc = zip([time.time()] * len(urls), titles, urls, documents)
                    predefined_table[keywords] = consolidated_doc
                print predefined_table
                self.agentObj.build_table(predefined_table)

            rec_docs, rec_urls, rec_titles = self.agentObj.performKNN(self.all_documents, self.all_urls, self.all_titles, cmd_keywords, cmd_k)
            print 'The Top ' + str(cmd_k) + ' jobs are:\n'
            print rec_urls

            try:
                html_code = '\n'.join(self.write_html(rec_titles, rec_urls))
                html_file = open("hw2_knn.html", "w")
                html_file.write(html_code)
                html_file.close()
                print "HTML File Written: hw2_knn.html"
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

            html_file = open("hw2_clusters.html", "w")
            for j, c in enumerate(clusters):
                print("cluster %d:" % j)
                for i in c:
                    print("\t%s" % clustering_documents[i])
                    printURLs.append(clustering_urls[i])
                    printTitles.append(clustering_titles[i])
                try:
                    html_code = '\n'.join(self.write_html_cluster(printTitles, printURLs, str(j)))

                    html_file.write(html_code)
                    print "HTML File Written: hw2_clusters.html"
                except Exception:
                    print "Writing to HTML file failed in clustering mode."
            html_file.close()
        except Exception:
            print "Error while clustering"
