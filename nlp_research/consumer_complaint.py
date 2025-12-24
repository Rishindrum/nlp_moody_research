from bertopic import BERTopic
# from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# df = pd.read_csv("D:/Documents/MachineLearning/nlp_research/complaints-2025-12-04_13_53.csv") #for quick running
df = pd.read_csv("D:/Documents/MachineLearning/nlp_research/complaints-2025-12-04_13_52.csv")


df.dropna(subset=['Consumer complaint narrative'],inplace=True)
docs = df['Consumer complaint narrative'].tolist()
df['Product'].fillna("Unknown Product", inplace=True)
products = df['Product'].tolist()
df['Tags'].fillna("Standard", inplace=True)
tags = df['Tags'].tolist()

print(df.head())
target_sizes = [15, 30, 60]
# target_sizes = [5]

# Experiment wth diff min_topic_sizes
for size in target_sizes:
    print(f"\n==========================================")
    print(f"RUNNING ANALYSIS: min_topic_size = {size}")
    print(f"==========================================")

    topic_model = BERTopic(min_topic_size=size, verbose=False) 
    topics, probs = topic_model.fit_transform(docs)
    
    topic_info = topic_model.get_topic_info()
    
    # -1 is the outlier topic. 
    outlier_row = topic_info[topic_info['Topic'] == -1]
    outlier_count = outlier_row['Count'].values[0] if not outlier_row.empty else 0
    
    # Total topics (excluding the outlier topic)
    real_topic_count = len(topic_info) - 1 if not outlier_row.empty else len(topic_info)
    
    print(f"1. Total Topics Found:   {real_topic_count}")
    print(f"2. Outlier Documents:    {outlier_count} ({(outlier_count/len(docs))*100:.2f}% of data)")
    
    print("\n3. Top 3 Topics:")
    top_topics = topic_model.get_topic_info().head(4)
    for index, row in top_topics.iterrows():
        if row['Topic'] == -1: continue # Skip outlier 
        print(f"   Topic {row['Topic']}: {row['Name']}")

    # Visualization by saving as HTML
    topics_per_product = topic_model.topics_per_class(docs, classes=products)
    fig = topic_model.visualize_topics_per_class(topics_per_product, top_n_topics=10)
    fig.write_html(f"topics_by_product_size_{size}.html")
    topics_per_tag = topic_model.topics_per_class(docs, classes=tags)
    fig = topic_model.visualize_topics_per_class(topics_per_tag, top_n_topics=10)
    fig.write_html(f"topics_by_tag_size_{size}.html")



    # Next Steps: data cleaning and pre processing of data, how to treat legal clause, and look more into pain points