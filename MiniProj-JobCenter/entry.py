import argparse as ap
from environment import Environment


if __name__ == "__main__":
    obj = Environment()
    # Required arguments k and w for K-Value and keyword sequence
    parser = ap.ArgumentParser(description='Career and Job Agent')
    parser.add_argument('-k', '--k_Val', help='Value of k, #top_jobs_required', required=True)
    parser.add_argument('-w', '--keywords', help='List of keywords', nargs='+', required=True)
    args = parser.parse_args()

    cmd_keywords = args.keywords
    cmd_keywords = " ".join(cmd_keywords)
    cmd_keywords = cmd_keywords.lower()
    print cmd_keywords

    cmd_k = int(args.k_Val)
    # While loop to persistently run the application until user enters a quit signal.
    # In our case it is any key other that 1
    while 1:
        obj.main(cmd_keywords, cmd_k)
        choice = raw_input('Select a choice: \n[Press 1] - New Keyword Search'
                           '\n[Press Any Other Key] - Quit The Application\n')
        if choice == '1':
            cmd_keywords = raw_input('Enter the new keywords\n').lower()
            cmd_k = input('Enter the value of k\n')
        else:
            quit()
    # Multi threading failed to scrape the jobs in parallel. I had to wait until the scraping finished.
    # scraping_thread = threading.Thread(target=obj.scrape_jobs,args=(predefined_keywords,))
    # scraping_thread.start()
    # while(scraping_thread.isAlive()):
    #     print "a."
    # print all_documents
    # print all_urls


# REFERENCES:
# https://stackoverflow.com/questions/25674169/how-does-the-list-comprehension-to-flatten-a-python-list-work
# https://gist.github.com/larsmans/4952848 --> Reference for K Means Clustering
# http://billchambers.me/tutorials/2014/12/22/cosine-similarity-explained-in-python.html
# |--> Cosine similarity computation, Bill Chambers, 22 December 2014
