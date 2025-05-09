import streamlit as st
from textblob import TextBlob
import pandas as pd
import emoji
from bs4 import BeautifulSoup
from urllib.request import urlopen
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (run once)
nltk.download('stopwords')

# Fetch Text From Url
@st.cache_data
def get_text(raw_url):
    try:
        page = urlopen(raw_url)
        soup = BeautifulSoup(page, 'html.parser')
        fetched_text = ' '.join(map(lambda p:p.text, soup.find_all('p')))
        return fetched_text
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return ""

def generate_wordcloud(text):
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         stopwords=stop_words,
                         min_font_size=10).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def plot_sentiment_distribution(sentiments):
    fig = px.histogram(sentiments, nbins=20, 
                      title='Sentiment Polarity Distribution',
                      labels={'value': 'Sentiment Polarity'})
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig)

def analyze_emojis(text):
    emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
    if emoji_list:
        emoji_counts = Counter(emoji_list)
        st.subheader("Emoji Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Top Emojis Found:")
            for emj, count in emoji_counts.most_common(5):
                st.write(f"{emj} - {count} times")
        with col2:
            fig = px.pie(names=list(emoji_counts.keys()), 
                        values=list(emoji_counts.values()),
                        title='Emoji Distribution')
            st.plotly_chart(fig)
    else:
        st.info("No emojis found in the text")

def main():
    """Enhanced Sentiment Analysis Emoji App"""
    st.set_page_config(page_title="Sentiment Analysis Pro", layout="wide")
    
    st.title("📊 Sentiment Analysis Pro")
    st.markdown("""
    <style>
    .big-font { font-size:18px !important; }
    </style>
    """, unsafe_allow_html=True)

    activities = ["Sentiment Analysis", "About"]
    choice = st.sidebar.selectbox("Menu", activities)

    if choice == 'Sentiment Analysis':
        st.subheader("🔍 Text Sentiment Analysis")
        st.write(emoji.emojize('Analyze text sentiment with emoji reactions :red_heart:'))
        
        raw_text = st.text_area("Enter Your Text", "I love this amazing app! 😊", height=150)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Analyze Sentiment", help="Click to analyze sentiment"):
                if raw_text.strip() == "Type Here" or not raw_text.strip():
                    st.warning("Please enter some text to analyze")
                else:
                    with st.spinner('Analyzing...'):
                        blob = TextBlob(raw_text)
                        result = blob.sentiment
                        
                        # Sentiment result
                        st.success("Analysis Complete!")
                        st.metric("Polarity Score", f"{result.polarity:.2f}", 
                                 help="Range from -1 (negative) to +1 (positive)")
                        st.metric("Subjectivity Score", f"{result.subjectivity:.2f}", 
                                 help="0 (objective) to 1 (subjective)")
                        
                        # Emoji reaction
                        if result.polarity > 0.3:
                            st.write(emoji.emojize("😊 Positive Sentiment"))
                        elif result.polarity < -0.3:
                            st.write(emoji.emojize("😞 Negative Sentiment"))
                        else:
                            st.write(emoji.emojize("😐 Neutral Sentiment"))
                        
                        # Sentiment gauge
                        fig = px.bar(x=[result.polarity], y=["Sentiment"], 
                                    orientation='h', 
                                    range_x=[-1, 1],
                                    title='Sentiment Polarity Meter',
                                    color_discrete_sequence=["green" if result.polarity > 0 
                                                           else "red" if result.polarity < 0 
                                                           else "gray"])
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if raw_text.strip() != "Type Here" and raw_text.strip():
                st.subheader("Text Insights")
                generate_wordcloud(raw_text)
                analyze_emojis(raw_text)

    
    elif choice == 'About':
        st.subheader("📝 About Sentiment Analysis Pro")
        st.markdown("""
        This enhanced app provides:
        - **Sentiment analysis** with emoji reactions
        """)
        st.markdown('<p class="big-font">Built with ❤️ using Streamlit, TextBlob, and Emoji</p>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        st.write("For educational purposes - demonstrating advanced NLP techniques")

if __name__ == '__main__':
    main()