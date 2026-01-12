import pandas as pd
from sentence_transformers import SentenceTransformer, util
import nltk
import re
import os
import consumer_complaint as cc
import torch

nltk.download('punkt')

def extract_ai_sentences_with_anchors(df, column='Consumer complaint narrative'):

    df = cc.data_cleaning(df, column)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    anchor_map = {
        'Agency': [
            "I could not reach a human representative.",
            "There was no way to speak to a real person.",
            "I was unable to contact anyone directly.",
            "Everything was handled without human interaction.",
            "I was blocked from speaking with a person."
        ],
        'Blame (Self-blame / Individual responsibility)': [
            "I may have made a mistake.",
            "This could be my fault.",
            "I did something wrong when submitting my information.",
            "I may be responsible for what happened.",
            "This issue may have resulted from something I did."
        ],
        'Blame (External (AI / system responsibility))': [
            "AI made an error.",
            "The algorithmic process failed.",
            "AI incorrectly handled my information.",
            "The system made an error in how it used my data.",
            "The automated system did not work properly."
        ],
        'Explanation / Transparency': [
            "I was not told why this decision was made.",
            "There was no explanation for the outcome.",
            "I don’t understand the reason for this decision.",
            "The decision was made without any explanation.",
            "The reasoning behind this decision was unclear."
        ],
        'AI / Digital Literacy': [
            "I am not familiar with how AI systems make decisions.",
            "I don’t understand how algorithms are used to make decisions.",
            "I am not sure how machine-learning systems evaluate customer information.",
            "I am unfamiliar with how large amounts of data are used to make decisions.",
            "I do not understand the logic behind algorithmic decision-making.",
            "I don’t know how my data is used in automated decisions."
        ],
        'Morality (Fairness/legitimacy)': [
            "This decision was unjust.",
            "I was treated imporperly by the system.",
            "This outcome was not justified.",
            "I was wrongfully denied.",
            "The system handled my case in an innapropriate manner.",
            "This decision lacks legitimacy."
        ],
        'Emotional Valence': [
            "This was very frustrating.",
            "I am upset about how this was handled.",
            "I feel helpless dealing with this system.",
            "This process has caused me significant stress.",
            "I am disappointed with how this was resolved.",
            "I feel overwhelmed by this situation."
        ]
    }
    
    # Keep track of anchors and their categories
    all_anchors = []
    anchor_categories = []
    for category, sentences in anchor_map.items():
        for sent in sentences:
            all_anchors.append(sent)
            anchor_categories.append(category)
            
    # Get Embeddings for all acnhor sentences
    anchor_embeddings = model.encode(all_anchors, convert_to_tensor=True)

    print("---- Finished Encoding Anchors ----")
    
    results = []
        
    # Go through all complaints
    for idx, row in df.iterrows():
        text = str(row[column])

        # 5 characters probably dont have enough info
        if len(text) < 5: continue
            
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Embed sentences
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        
        # Compute Cos Similarity Matrix of Sentences and Anchors
        cosine_scores = util.cos_sim(sentence_embeddings, anchor_embeddings)
        
        # Threshold: Need to work on tweaking this, kept at 0 for now, but could increase later
        threshold = 0
        
        for i, sent in enumerate(sentences):
            # Find the BEST 3 anchors for this sentence
            sim_scores = cosine_scores[i]
            top_3 = torch.topk(sim_scores, k=3)
            
            idx_1 = top_3.indices[0].item()
            idx_2 = top_3.indices[1].item()
            idx_3 = top_3.indices[2].item()
            
            val_1 = top_3.values[0].item()
            val_2 = top_3.values[1].item()
            val_3 = top_3.values[2].item()
            
            margin_1_2 = val_1 - val_2
            margin_2_3 = val_2 - val_3
            
            
            results.append({
                'Complaint ID': row['Complaint ID'],
                'Consumer Group': row['Tags'],
                'Product': row['Product'],
                'Extracted Sentence': sent,
                
                'Category_1': anchor_categories[idx_1],
                'Anchor Text 1': all_anchors[idx_1],
                'Score_1': round(val_1, 4),
                
                'Category_2': anchor_categories[idx_2],
                'Anchor Text 2': all_anchors[idx_2],
                'Score_2': round(val_2, 4),

                'Category_3': anchor_categories[idx_3],
                'Anchor Text 3': all_anchors[idx_3],
                'Score_3': round(val_3, 4),
                
                'Margin (1-2)': round(margin_1_2, 4),
                'Margin (2-3)': round(margin_2_3, 4),
            })
                
    sentence_embeddings_df = pd.DataFrame(results)
    print(f"Extraction Complete. Found {len(sentence_embeddings_df)} sentences matching the anchors/templates.")
    
    return sentence_embeddings_df

# Directory where this file resides to get the csv files
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_1 = os.path.join(script_dir, "complaints-2025-12-04_13_52.csv")
file_path_2 = os.path.join(script_dir, "complaints-2025-12-04_13_53.csv")

df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)
df = pd.concat([df1, df2], ignore_index=True)
df_extracted = extract_ai_sentences_with_anchors(df)
df_extracted.to_csv("extracted_ai_narratives.csv", index=False)