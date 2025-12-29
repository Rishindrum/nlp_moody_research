from bertopic import BERTopic
import pandas as pd
import re
from collections import Counter
import nltk
import os
from nltk.util import ngrams

# nltk.download('punkt_tab')


def data_cleaning(df, column):
    print("---- Data Cleaning Started  ----")
    cur_size = starting_size = len(df)
    print(f"Starting dataframe size: {starting_size}")

    # N/A Handling and Lowercase
    df.dropna(subset=[column],inplace=True) # No need to keep data without any complaint
    df['Product'].fillna("Unknown Product", inplace=True)
    df['Tags'].fillna("Standard", inplace=True)
    print(" -- N/A Removal --")
    print(f"Removed {cur_size - len(df)} null entries from dataframe")
    cur_size = len(df)

    # Standard Case, Punctuation, and Extra Spaces Removal so we avoid treating words that are the same with just some punctuation different
    df[column] = df[column].str.replace(r'[^\w\s]', '', regex=True) # Remove punctuation
    df[column] = df[column].str.replace(r'\s+', ' ', regex=True).str.strip() # Fix whitespace
    df[column] = df[column].str.lower()

    # XXXX Redacted Placeholder handling, two or more x letters in English is very unlikely so I chose two Xs to replace
    # There are exceptions like Exxon or XXL but they're not necessarily words and they also aren't in the dataset
    num_placeholders = df[column].str.count(r'x{2,}').sum()
    df[column] = df[column].apply(lambda x: re.sub(r'x{2,}', '', x))
    print(" -- XX Redacted Placeholder Removal --")
    print(f"Detected and substituted {num_placeholders} instances of redacted 'XXXX' placeholders.")



    # Boilerplate / Template Removal
    # Look for n-word segments in the whole text, playing around with different length segments
    # Choose a spot where anything less identfies segments that felt like normal sentences

    # N-Selection
    # 5- word was too small, phrases were normal like "recently recieved a copy of" or "violation of the law"
    # 10-word looked good, 'your company is in clear violation of the law pursuant' or 'between the consumer and the person making the report the' seems unnatural in normal sentences
    # Anything more was getting too specific like specific parts of the law: isinclearviolationofthelawpursuant15usc1681a2aiexclusions or yourcompanyisinclearviolationofthelawpursuant15usc1681a2aiexclusionsexceptasprovided
    
    # Threshold Selection
    # 25 had certain phrases that couldve been just normal language caught like 'i have received a copy of my consumer reports'
    # 50 has a lot of legal citations about 15 usc and a paypal glitch: paypal usd py usd py usd sent paypal...
    # Anything mroe and we miss out on things like the paypal glitch and most law templates
    
    print(" -- Boilerplate Template Removal -- ")
    full_text =  " ".join(df[column].tolist())
    word_tokens = nltk.word_tokenize(full_text)
    # n_sizes = [5, 10, 15, 20]
    n_sizes = [10]
    # thresholds = [25, 50, 100, 500]
    thresholds = [50]
    template_phrases = []

    for n in n_sizes:
        # print(f"SIZE: {n}")
        # print("-----------------------------------------------------------------------")
        sequences = ngrams(word_tokens, n)
        seqCounter = Counter(sequences)
        for threshold in thresholds:
            # print(f"Threshold: {threshold}")
            # print("--------------------------------------------------------------------")
            for phrase in seqCounter.keys():
                if seqCounter[phrase] >= threshold:
                    string_phrase = " ".join(phrase)
                    template_phrases.append(string_phrase)

    print(" Below are all the phrases to remove.  ")
    print(template_phrases)
    print(f"   Found {len(template_phrases)} boilerplate/template phrases to remove.")

    for phrase in template_phrases:
        df[column] = df[column].str.replace(phrase, '', regex=False)

    cur_size = len(df)
    df = df[df[column].str.len() > 3].copy()
    print(f" Dropping {cur_size - len(df)} complaints without anything left (fully templated and/or redacted)")

    print(f" Data Cleaning completed. Dropped a total of {starting_size - len(df)} rows during cleaning.")

    return df

def bertopic_analysis(dataframes, topic_list):

    # Combine dataframes
    df = pd.concat(dataframes, ignore_index=True)
    df = data_cleaning(df, 'Consumer complaint narrative')

    products = df['Product'].tolist()
    tags = df['Tags'].tolist()
    docs = df['Consumer complaint narrative'].tolist()

    print(df.head())
    target_sizes = [15]
    # target_sizes = [15, 30, 60]

    # Experiment wth diff min_topic_sizes
    for size in target_sizes:
        print(f"\n==========================================")
        print(f"RUNNING ANALYSIS: min_topic_size = {size}")
        print(f"==========================================")

        topic_model = BERTopic(
            seed_topic_list=topic_list,
            min_topic_size=size,
            verbose=True
        )

        topic_model.fit_transform(docs)
        
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
        fig = topic_model.visualize_topics_per_class(topics_per_tag, top_n_topics=30)
        fig.write_html(f"topics_by_tag_size_{size}.html")

def analyze_legal_language(dataframes, column):

    df = pd.concat(dataframes, ignore_index=True)
    df.dropna(subset=[column],inplace=True) # No need to keep data without any complaint
    df['Product'].fillna("Unknown Product", inplace=True)
    df['Tags'].fillna("Standard", inplace=True)

    print("--- Legal & Regulatory Language Analysis ---")
    
    
    legal_sections = {
        'Specific_Laws': [
            r"fair credit reporting act", r"fcra",
            r"equal credit opportunity act", r"ecoa",
            r"fair debt collection practices act", r"fdcpa",
            r"dodd.?frank", # to match with - or space
            r"15 usc", r"15 us code",
            r"section \d+"
        ],
        'Legal_Phrasing': [
            r"violation of",
            r"pursuant to",
            r"in accordance with",
            r"noncompliance",
            r"entitled to",
            r"unlawful",
            r"statutory",
            r"federal law",
            r"state law",#
            r"liability",
            r"defamation"
        ]
    }

    all_complaints = df[column].astype(str).str.lower()
    
    # Create a column for "Mentioned Specific Law" and "Used Legal Phrasing"
    
    for category, patterns in legal_sections.items():
        #Match with any of the regex phrases
        full_regex = "|".join(patterns)
        
        # Create column
        col_name = f"Has_{category}"
        df[col_name] = all_complaints.str.contains(full_regex, regex=True)

    df['Has_Any_Legal'] = df['Has_Specific_Laws'] | df['Has_Legal_Phrasing']

    
    # Calculate Count (N) and Mean (%) for the 'Has_Any_Legal' column
    summary_table = df.groupby('Tags')[['Has_Any_Legal']].agg(['count', 'sum', 'mean'])
    
    # Store and format summary values 
    summary_table.columns = ['Total Narratives', 'Count with Legal Lang', '% Frequency']
    summary_table['% Frequency'] = (summary_table['% Frequency'] * 100).round(2)
    summary_table = summary_table.sort_values('% Frequency', ascending=False)
    
    print("\n-------- Table: Usage of Legal Language by Group ------")
    print(summary_table)
    print("\n--- Breakdown by Category ---")
    breakdown = df.groupby('Tags')[['Has_Specific_Laws', 'Has_Legal_Phrasing']].mean() * 100
    print(breakdown)

    return summary_table, df

#Directory where this file resides to get the csv files
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_1 = os.path.join(script_dir, "complaints-2025-12-04_13_52.csv")
file_path_2 = os.path.join(script_dir, "complaints-2025-12-04_13_53.csv")

seed_topic_list = [
    ["chatbot", "automated", "loop", "agent", "human", "stuck"],  #Support Loop
    ["algorithm", "denied", "score", "computer", "system", "decision"], #Unsure/Black Box
    ["verification", "upload", "id", "scan", "camera", "face"],   #Identity
    ["fraud", "scam", "unauthorized", "hack", "money"]             #Fraud
]

df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)
analyze_legal_language([df1, df2], 'Consumer complaint narrative')
# bertopic_analysis([df1, df2], seed_topic_list)
