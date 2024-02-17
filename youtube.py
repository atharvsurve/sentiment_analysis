from flask import Flask , render_template , request
import matplotlib.pyplot as plt 
from googleapiclient.discovery import build
import pandas as pd
# import libraries
import pandas as pd
from io import BytesIO
import base64

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

# download nltk corpus (first time only)


api_key = 'AIzaSyCYIeLeKeChEaYJ6KEtg-Kuy4ryeZkgCpk'
app = Flask(__name__)




        
def video_comments(video_id):
    # Empty lists for storing comments and replies
    comments_list = []
    replies_list = []

    # Creating YouTube resource object
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Set up the initial request
    video_response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id
    ).execute()

    # Iterate through video response
    while video_response:
        # Extracting required info from each result object
        for item in video_response['items']:
            # Extracting comments
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments_list.append(comment)

            # Counting the number of replies to the comment
            replycount = item['snippet']['totalReplyCount']

            # If there are replies, extract them
            if replycount > 0:
                # Iterate through all replies
                for reply in item['replies']['comments']:
                    # Extract reply
                    reply_text = reply['snippet']['textDisplay']
                    # Append reply to the list of replies
                    replies_list.append(reply_text)

        # Check for the next page
        if 'nextPageToken' in video_response:
            next_page_token = video_response['nextPageToken']
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=next_page_token
            ).execute()
        else:
            break

    # Create a DataFrame from the lists
    df = pd.DataFrame({'comments': comments_list})

    # Save the DataFrame to a CSV file
    


    return df

# Enter the video ID
# video_id = 'k0h_55Odqyc'

# 'iWgzyS8h3Iw' an example of bad comments 
# 'sJa9EZHAlI4' an exmple for good comments 

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

analyzer = SentimentIntensityAnalyzer()

# Create a function to apply sentiment analysis
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 'Positive' if scores['compound'] >= 0.05 else 'Negative' if scores['compound'] <= -0.05 else 'Neutral'
    return sentiment






       
@app.route('/' , methods = ['GET' , 'POST'])   
def hello_world():
     if request.method == 'POST':
         Youtube_url = request.form['youtube_url']
         index = Youtube_url.find('=')
         video_id = Youtube_url[index+1: ]
         
         dframe = video_comments(video_id)
         dframe['comments'] = dframe['comments'].apply(preprocess_text)
         dframe['sentiment'] = dframe['comments'].apply(get_sentiment)
         sentiment_counts = dframe['sentiment'].value_counts()
         
         plt.pie(sentiment_counts, labels=['Positive', 'Non-Positive' , 'Neutral'], colors=['green', 'red' , 'yellow'], autopct='%1.1f%%', startangle=90)
         
         buffer = BytesIO()
         plt.savefig(buffer, format='png')
         buffer.seek(0)
         plt.close()

        # Convert the BytesIO buffer to a base64-encoded image
         plot_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
         return render_template('index.html', plot_image=plot_image)

     return render_template('index.html')


         



         
        

    

if __name__ == '__main__' :
    app.run(port = 3000 , debug = True)
    