import re
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

class WhatsAppAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.messages = []
        self.df = None
        
    def parse_chat(self):
        """Parse WhatsApp chat export"""
        pattern = r'\[(\d{2}/\d{2}/\d{2}),\s(\d{1,2}:\d{2}:\d{2}\s[AP]M)\]\s([^:]+):\s(.+)'
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        current_msg = None
        for line in lines:
            match = re.match(pattern, line)
            if match:
                if current_msg:
                    self.messages.append(current_msg)
                
                date_str, time_str, sender, message = match.groups()
                timestamp_str = f"{date_str}, {time_str}"
                timestamp = datetime.strptime(timestamp_str, '%d/%m/%y, %I:%M:%S %p')
                
                msg_text = message.strip()
                if 'omitted' in msg_text:
                    msg_text = '<Media omitted>'
                
                current_msg = {
                    'timestamp': timestamp,
                    'sender': sender.strip().replace('~', '').strip(),
                    'message': msg_text,
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.strftime('%A'),
                    'date': timestamp.date()
                }
            elif current_msg and line.strip():
                current_msg['message'] += ' ' + line.strip()
        
        if current_msg:
            self.messages.append(current_msg)
        
        self.df = pd.DataFrame(self.messages)
        
        if len(self.messages) > 0:
            print(f"📱 Parsed {len(self.messages)} messages from {len(self.df['sender'].unique())} participants")
        else:
            print("⚠️ No messages found. Please check the chat file format.")
        return self.df
    
    def analyze_activity(self):
        """Analyze participant activity"""
        print("\n🏆 PARTICIPANT LEADERBOARD")
        print("=" * 50)
        
        activity_stats = []
        for sender in self.df['sender'].unique():
            sender_msgs = self.df[self.df['sender'] == sender]
            total_msgs = len(sender_msgs)
            avg_length = sender_msgs['message'].str.len().mean()
            
            activity_stats.append({
                'name': sender,
                'total_messages': total_msgs,
                'avg_message_length': avg_length,
                'most_active_hour': sender_msgs.groupby('hour').size().idxmax(),
                'favorite_day': sender_msgs.groupby('day_of_week').size().idxmax()
            })
        
        activity_df = pd.DataFrame(activity_stats).sort_values('total_messages', ascending=False)
        
        print("\n🗣️ TOP CHATTERS:")
        for i, row in activity_df.head(5).iterrows():
            emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][min(i, 4)]
            print(f"{emoji} {row['name']}: {row['total_messages']} messages")
            print(f"   → Avg length: {row['avg_message_length']:.0f} chars")
            print(f"   → Peak hour: {row['most_active_hour']}:00")
            print(f"   → Favorite day: {row['favorite_day']}")
        
        print("\n🤐 QUIETEST MEMBERS:")
        for _, row in activity_df.tail(3).iterrows():
            print(f"😴 {row['name']}: {row['total_messages']} messages")
        
        return activity_df
    
    def analyze_sentiment(self):
        """Analyze sentiment of messages"""
        print("\n😊 SENTIMENT ANALYSIS")
        print("=" * 50)
        
        sentiments = []
        for _, row in self.df.iterrows():
            if not row['message'].startswith('<Media omitted>'):
                blob = TextBlob(row['message'])
                sentiments.append({
                    'sender': row['sender'],
                    'message': row['message'],
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
        
        sentiment_df = pd.DataFrame(sentiments)
        
        sender_sentiment = sentiment_df.groupby('sender').agg({
            'polarity': 'mean',
            'subjectivity': 'mean'
        }).round(3)
        
        print("\n😄 MOST POSITIVE PEOPLE:")
        top_positive = sender_sentiment.nlargest(3, 'polarity')
        for sender, row in top_positive.iterrows():
            emoji = '😄' if row['polarity'] > 0.3 else '🙂'
            print(f"{emoji} {sender}: {row['polarity']:.3f} positivity score")
        
        print("\n😔 MOST NEGATIVE/NEUTRAL PEOPLE:")
        top_negative = sender_sentiment.nsmallest(3, 'polarity')
        for sender, row in top_negative.iterrows():
            emoji = '😔' if row['polarity'] < -0.1 else '😐'
            print(f"{emoji} {sender}: {row['polarity']:.3f} positivity score")
        
        print("\n🎭 MOST DRAMATIC (subjective):")
        top_subjective = sender_sentiment.nlargest(3, 'subjectivity')
        for sender, row in top_subjective.iterrows():
            print(f"🎭 {sender}: {row['subjectivity']:.3f} subjectivity score")
        
        return sentiment_df
    
    def find_hot_topics(self):
        """Find hot topics using TF-IDF and topic modeling"""
        print("\n🔥 HOT TOPICS")
        print("=" * 50)
        
        messages = [msg for msg in self.df['message'] if not msg.startswith('<Media omitted>')]
        
        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3
        )
        
        tfidf_matrix = vectorizer.fit_transform(messages)
        feature_names = vectorizer.get_feature_names_out()
        
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(tfidf_matrix)
        
        print("\n📊 TOP 5 DISCUSSION THEMES:")
        for idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"\n🏷️ Theme {idx + 1}:")
            print(f"   {', '.join(top_words[:5])}")
        
        all_words = ' '.join(messages).lower()
        word_freq = Counter(all_words.split())
        
        common_topics = [
            'modi', 'bjp', 'congress', 'trump', 'usa', 'china', 'pakistan', 'election', 'politics',
            'stock', 'crypto', 'bitcoin', 'gold', 'investment', 'nifty', 'sensex', 'market',
            'bangalore', 'mumbai', 'delhi', 'chennai', 'hyderabad', 'karnataka', 'maharashtra',
            'work', 'job', 'salary', 'office', 'meeting', 'wfh', 'tech', 'software',
            'movie', 'bollywood', 'cricket', 'ipl', 'kohli', 'dhoni', 'netflix',
            'food', 'biryani', 'restaurant', 'zomato', 'swiggy', 'lunch', 'dinner',
            'travel', 'vacation', 'goa', 'kerala', 'flight', 'uber', 'ola',
            'house', 'rent', 'flat', 'traffic', 'metro', 'weather', 'rain', 'monsoon',
            'festival', 'diwali', 'holi', 'wedding', 'family', 'friends', 'party', 'weekend',
            'covid', 'vaccine', 'lockdown', 'news', 'plan', 'tomorrow', 'tonight'
        ]
        
        print("\n🗣️ MOST TALKED ABOUT:")
        topic_counts = {}
        for topic in common_topics:
            count = sum(1 for msg in messages if topic.lower() in msg.lower())
            if count > 0:
                topic_counts[topic] = count
        
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {topic.capitalize()}: mentioned {count} times")
        
        return feature_names, lda
    
    def find_polarizing_topics(self):
        """Find topics with most varied sentiment"""
        print("\n⚡ POLARIZING TOPICS")
        print("=" * 50)
        
        topic_sentiments = defaultdict(list)
        topics_to_check = ['trump', 'biden', 'usa', 'china', 'pakistan', 'election', 'politics', 'modi', 
                          'bjp', 'congress', 'crypto', 'bitcoin', 'stock', 'market', 'investment', 'gold',
                          'movie', 'bollywood', 'cricket', 'kohli', 'ipl', 'work', 'salary', 'job',
                          'bangalore', 'mumbai', 'delhi', 'traffic', 'rent', 'food', 'restaurant',
                          'weekend', 'party', 'travel', 'weather', 'covid', 'vaccine', 'lockdown']
        
        for _, row in self.df.iterrows():
            msg_lower = row['message'].lower()
            for topic in topics_to_check:
                if topic in msg_lower and not row['message'].startswith('<Media omitted>'):
                    blob = TextBlob(row['message'])
                    topic_sentiments[topic].append(blob.sentiment.polarity)
        
        polarizing = []
        for topic, sentiments in topic_sentiments.items():
            if len(sentiments) > 2:
                std_dev = np.std(sentiments)
                polarizing.append((topic, std_dev, len(sentiments)))
        
        polarizing.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🔄 Topics with most varied opinions:")
        for topic, std, count in polarizing[:5]:
            print(f"   • {topic.capitalize()}: σ={std:.3f} ({count} mentions)")
        
        return polarizing
    
    def analyze_time_patterns(self):
        """Analyze time patterns"""
        print("\n⏰ TIME PATTERNS")
        print("=" * 50)
        
        hourly = self.df.groupby('hour').size()
        daily = self.df.groupby('day_of_week').size()
        
        peak_hour = hourly.idxmax()
        quiet_hour = hourly.idxmin()
        peak_day = daily.idxmax()
        quiet_day = daily.idxmin()
        
        print(f"\n🌟 Peak chat hour: {peak_hour}:00 ({hourly.max()} messages)")
        print(f"😴 Quietest hour: {quiet_hour}:00 ({hourly.min()} messages)")
        print(f"🗓️ Most active day: {peak_day} ({daily.max()} messages)")
        print(f"🤫 Quietest day: {quiet_day} ({daily.min()} messages)")
        
        if peak_hour >= 22 or peak_hour <= 2:
            print("\n🦉 You guys are night owls!")
        elif peak_hour >= 6 and peak_hour <= 9:
            print("\n🌅 Early bird squad!")
        
        return hourly, daily
    
    def find_conversation_starters(self):
        """Find who starts most conversations"""
        print("\n💬 CONVERSATION DYNAMICS")
        print("=" * 50)
        
        self.df['time_diff'] = self.df['timestamp'].diff()
        conversation_starters = []
        
        for idx, row in self.df.iterrows():
            if idx == 0 or row['time_diff'].total_seconds() > 3600:
                conversation_starters.append(row['sender'])
        
        starter_counts = Counter(conversation_starters)
        
        print("\n🚀 TOP CONVERSATION STARTERS:")
        for sender, count in starter_counts.most_common(3):
            print(f"   🗣️ {sender}: started {count} conversations")
        
        return starter_counts
    
    def generate_fun_awards(self):
        """Generate fun awards for group members"""
        print("\n🏆 FUN AWARDS")
        print("=" * 50)
        
        emoji_users = defaultdict(int)
        question_users = defaultdict(int)
        exclamation_users = defaultdict(int)
        media_users = defaultdict(int)
        
        for _, row in self.df.iterrows():
            sender = row['sender']
            msg = row['message']
            
            emoji_count = len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿🏴-🏿]', msg))
            emoji_users[sender] += emoji_count
            
            if '?' in msg:
                question_users[sender] += 1
            if '!' in msg:
                exclamation_users[sender] += 1
            if '<Media omitted>' in msg:
                media_users[sender] += 1
        
        print(f"😂 Emoji Champion: {max(emoji_users, key=emoji_users.get)}")
        print(f"❓ Question Master: {max(question_users, key=question_users.get)}")
        print(f"❗ Excitement King/Queen: {max(exclamation_users, key=exclamation_users.get)}")
        if media_users:
            print(f"📸 Media Mogul: {max(media_users, key=media_users.get)}")
        
        avg_msg_length = self.df.groupby('sender')['message'].apply(lambda x: x.str.len().mean())
        print(f"📜 Essay Writer: {avg_msg_length.idxmax()} ({avg_msg_length.max():.0f} chars avg)")
        print(f"🔤 Short & Sweet: {avg_msg_length.idxmin()} ({avg_msg_length.min():.0f} chars avg)")
        
        return {
            'emoji_champion': max(emoji_users, key=emoji_users.get),
            'question_master': max(question_users, key=question_users.get),
            'excitement_person': max(exclamation_users, key=exclamation_users.get)
        }
    
    def create_visualizations(self):
        """Create fun visualizations"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        activity_by_sender = self.df['sender'].value_counts().head(10)
        axes[0, 0].barh(activity_by_sender.index, activity_by_sender.values, color='skyblue')
        axes[0, 0].set_xlabel('Number of Messages')
        axes[0, 0].set_title('🏆 Top 10 Most Active Members', fontsize=14, fontweight='bold')
        axes[0, 0].invert_yaxis()
        
        hourly_activity = self.df.groupby('hour').size()
        axes[0, 1].plot(hourly_activity.index, hourly_activity.values, marker='o', color='coral', linewidth=2)
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Number of Messages')
        axes[0, 1].set_title('⏰ Chat Activity Throughout the Day', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_activity = self.df.groupby('day_of_week').size().reindex(day_order)
        axes[1, 0].bar(daily_activity.index, daily_activity.values, color='lightgreen')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Number of Messages')
        axes[1, 0].set_title('📅 Weekly Activity Pattern', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        messages_text = ' '.join([msg for msg in self.df['message'] if not msg.startswith('<Media omitted>')])
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             colormap='viridis', max_words=50).generate(messages_text)
        axes[1, 1].imshow(wordcloud, interpolation='bilinear')
        axes[1, 1].set_title('☁️ Most Common Words', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('/Users/pcadabam/Downloads/buddies/whatsapp_analysis.png', dpi=150, bbox_inches='tight')
        print("\n📊 Visualizations saved as 'whatsapp_analysis.png'")
        
        return fig
    
    def generate_summary_report(self):
        """Generate a fun summary report"""
        print("\n" + "=" * 60)
        print("🎉 APARTMENT GROUP CHAT PERSONALITY REPORT 🎉")
        print("=" * 60)
        
        total_days = (self.df['timestamp'].max() - self.df['timestamp'].min()).days
        msgs_per_day = len(self.df) / max(total_days, 1)
        
        print(f"\n📈 QUICK STATS:")
        print(f"   • Total messages: {len(self.df):,}")
        print(f"   • Active members: {len(self.df['sender'].unique())}")
        print(f"   • Time period: {total_days} days")
        print(f"   • Average messages per day: {msgs_per_day:.1f}")
        
        if msgs_per_day > 100:
            print("\n🔥 This group is ON FIRE! You guys never stop talking!")
        elif msgs_per_day > 50:
            print("\n💬 Pretty chatty group! Good vibes all around!")
        else:
            print("\n😌 Nice and chill group dynamics!")

def main():
    print("🎊 WHATSAPP GROUP CHAT ANALYZER 🎊")
    print("=" * 60)
    
    analyzer = WhatsAppAnalyzer('/Users/pcadabam/Downloads/buddies/_chat.txt')
    
    analyzer.parse_chat()
    
    analyzer.generate_summary_report()
    
    activity_df = analyzer.analyze_activity()
    
    sentiment_df = analyzer.analyze_sentiment()
    
    features, lda = analyzer.find_hot_topics()
    
    polarizing = analyzer.find_polarizing_topics()
    
    hourly, daily = analyzer.analyze_time_patterns()
    
    starters = analyzer.find_conversation_starters()
    
    awards = analyzer.generate_fun_awards()
    
    fig = analyzer.create_visualizations()
    
    print("\n" + "=" * 60)
    print("🎭 Analysis Complete! Share these fun insights with your group!")
    print("=" * 60)

if __name__ == "__main__":
    main()