from googleapiclient.discovery import build
from typing import  Any

query = 'Allen Crags'
'''
This is too slow
for j in search(query, tld="co.in", num=2, stop=2, pause=0.1):
    print(j)

    reqs = requests.get(j)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    # soup = BeautifulSoup((urlopen(j)))
    # print("Title of the website is: ", soup.title.get_text())
    print(soup.findAll("title"))
    for title in soup.findAll("title"):
        print(title.get_text())


'''


class WebSearch:

    def __init__(self, cse_id ="c08f6c0abe4964877"):  # , api_key, cse_id
        """
        Init: connect with Google custom search api
        Parameters api
        ----------
        api_key: api key of custom search api
        cse_id: id of user's custom search
        """
        try:
            self.my_api_key = "AIzaSyBogbAxNrr0c39ChcCTHliSBiMHVSQ7mog"
            self.my_cse_id = cse_id
            self.service = build("customsearch", "v1", developerKey=self.my_api_key)
            self.res = []
        except ValueError:
            pass

    def search_result(self, search_term: str, top_n: int, **kwargs) :
        """
        crawl the top n results of searched item
        Parameters
        ----------
        search_term: input search string
        top_n: top n webpage
        kwargs: other parameter that may be used in cse()

        Returns none
        -------

        """
        if top_n <= 0:
            # print("N must be beyond 0!")
            pass
        start_num = 0
        iteration = 0
        while top_n - iteration * 10 >= 10:
            # print(start_num, iteration)
            cur_res = self.service.cse().list(q=search_term, cx=self.my_cse_id, num=10,
                                              start=start_num, **kwargs).execute()
            if 'items' in dict(cur_res).keys():
                self.res = self.res + cur_res['items']
            self.res = self.res + cur_res['items']
            iteration = iteration + 1
            start_num = 10 * iteration + 1
        # print(top_n - iteration*10, iteration, start_num)
        other_res = self.service.cse().list(q=search_term, cx=self.my_cse_id, num=top_n - iteration * 10,
                                            start=start_num, **kwargs).execute()
        if 'items' in dict(other_res).keys():
            self.res = self.res + other_res['items']
        return self.res


'''
web_search = WebSearch()
results = web_search.search_result(query, 10)
print(results)

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']
    
    
results = google_search(query, my_api_key, my_cse_id, num=2)
print(results)
for result in results:
    # pprint.pprint(result)
    print(result["title"], result["snippet"])

'''
