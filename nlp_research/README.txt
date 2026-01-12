Week 1:
First, I did the BERTopic analysis on the two datasets combined without any starting seed topic list. 
Dealt with null values and repalced no tag with Standard and no product with Unknown
Here, there were still issues with placeholders like xxx and lots of usage of templates and irrelevant financial terms.
Messed around with different cluster sizes and visualized the topic groups created by bertopic compared to the tags and products

Weeks 2 and 3:
Cleaned the dataset better by setting everything to lowercase, removing punctuation and spacing so the same words werent treated differently, etc.
Also removed the xxx placeholders with Regex logic, .apply, and re.sub()
Did boilerplate removal of n-gram sequences, finalizing with 10-gram sequences occuring 50 times within the dataset to be flagged as boilerplate/templates
Even after all this, however, the same problem of financial topics domianting persisted
Tried to add the seed_topic_list with identity, chat loops, fraud, etc. words but it still didnt make a big difference
Successfully kept track of which % complaints and were law/legal related by using regex and comparing to a list of words focusing on legal phrasing and specific laws

