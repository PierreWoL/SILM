from googleapiclient.discovery import (build)


class WebSearch:

    def __init__(self, cse_id=""):
        """
        Init: connect with Google custom search api
        Parameters api
        ----------
        api_key: api key of custom search api
        cse_id: id of user's custom search
        """
        try:
            self.my_api_key = ""
            self.my_cse_id = cse_id
            self.service = build("customsearch", "v1", developerKey=self.my_api_key)
            self.res = []
            self.contents = []
        except ValueError:
            pass

    def search_result(self, search_term: str, top_n: int, **kwargs):
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
        for res in self.res:
            content = {'title': "", "snippet": ""}
            if "title" in res.keys():
                content["title"] = res["title"]
            if "snippet" in res.keys():
                content["snippet"] = res["snippet"]
            self.contents.append(content)
        return self.contents


"""query = 'Anguilla'
web_search = WebSearch()
results = web_search.search_result(query, 5)
for result in results:
    # pprint.pprint(result)
    print(result["title"], result["snippet"])"""
