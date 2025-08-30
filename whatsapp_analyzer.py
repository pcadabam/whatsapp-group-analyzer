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
        # Try multiple patterns to support different WhatsApp export formats
        patterns = [
            # US formats with brackets (iOS common)
            r'\[(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}:\d{2}\s[AP]M)\]\s([^:]+):\s(.+)',  # [m/d/yy, h:mm:ss AM/PM]
            r'\[(\d{1,2}/\d{1,2}/\d{4}),\s(\d{1,2}:\d{2}:\d{2}\s[AP]M)\]\s([^:]+):\s(.+)',  # [m/d/yyyy, h:mm:ss AM/PM]
            r'\[(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}\s[AP]M)\]\s([^:]+):\s(.+)',  # [m/d/yy, h:mm AM/PM]
            
            # European formats with brackets
            r'\[(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}:\d{2}\s[AP]M)\]\s([^:]+):\s(.+)',  # [d/m/yy, h:mm:ss AM/PM]
            r'\[(\d{1,2}/\d{1,2}/\d{4}),\s(\d{1,2}:\d{2}:\d{2})\]\s([^:]+):\s(.+)',  # [d/m/yyyy, HH:mm:ss] 24hr
            r'\[(\d{1,2}\.\d{1,2}\.\d{2}),\s(\d{1,2}:\d{2})\]\s([^:]+):\s(.+)',  # [dd.mm.yy, HH:mm] German style
            
            # Standard formats without brackets (Android common)
            r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)',  # dd/mm/yyyy, HH:mm - Name: Message
            r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s[ap]m)\s-\s([^:]+):\s(.+)',  # dd/mm/yyyy, h:mm am/pm
            r'(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.+)',  # mm/dd/yy, h:mm AM/PM
            
            # Dash separator formats  
            r'(\d{1,2}-\d{1,2}-\d{2,4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)',  # dd-mm-yyyy, HH:mm
            r'(\d{1,2}-\d{1,2}-\d{2}),\s(\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.+)',  # mm-dd-yy, h:mm AM/PM
            
            # ISO/Asian formats
            r'(\d{4}-\d{2}-\d{2}),\s(\d{1,2}:\d{2}:\d{2})\s-\s([^:]+):\s(.+)',  # yyyy-mm-dd, HH:mm:ss
            r'\[(\d{4}-\d{2}-\d{2}),\s(\d{1,2}:\d{2}:\d{2})\]\s([^:]+):\s(.+)',  # [yyyy-mm-dd, HH:mm:ss]
            
            # Formats with different spacing
            r'(\d{1,2}/\d{1,2}/\d{2,4})\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)',  # No comma after date
            r'(\d{1,2}/\d{1,2}/\d{2,4}),(\d{1,2}:\d{2})-([^:]+):\s(.+)',  # No spaces
            
            # Dot separator formats (some European countries)
            r'(\d{1,2}\.\d{1,2}\.\d{2,4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)',  # dd.mm.yyyy, HH:mm
            r'(\d{1,2}\.\d{1,2}\.\d{2}),\s(\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.+)',  # dd.mm.yy, h:mm AM/PM
            
            # Formats with seconds
            r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\s-\s([^:]+):\s(.+)',  # dd/mm/yyyy, HH:mm:ss
            r'(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.+)',  # mm/dd/yy, h:mm:ss AM/PM
            
            # WhatsApp Business formats
            r'(\d{1,2}/\d{1,2}/\d{2,4})\sat\s(\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.+)',  # dd/mm/yyyy at h:mm AM/PM
            r'\[(\d{1,2}/\d{1,2}/\d{2,4})\sat\s(\d{1,2}:\d{2}\s[AP]M)\]\s([^:]+):\s(.+)',  # [dd/mm/yyyy at h:mm AM/PM]
            
            # Additional edge case formats
            r'(\d{1,2}_\d{1,2}_\d{2,4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)',  # Underscore separators
            r'(\d{4}/\d{2}/\d{2}),\s(\d{1,2}:\d{2}:\d{2})\s-\s([^:]+):\s(.+)',  # yyyy/mm/dd format
            r'(\d{1,2}/\d{1,2}),\s(\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.+)',  # Missing year
            r'(\d{1,2}/\d{1,2}/\d{2,4})\s@\s(\d{1,2}:\d{2}\s[AP]M)\s-\s([^:]+):\s(.+)',  # Using @ instead of at
            r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2}\.\d{3})\s-\s([^:]+):\s(.+)',  # With milliseconds
            
            # Formats with timezone
            r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s[AP]M\s[A-Z]{3})\s-\s([^:]+):\s(.+)',  # With timezone
            r'\[(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}:\d{2}\s[AP]M\s[A-Z]{3})\]\s([^:]+):\s(.+)',  # [date, time TZ]
        ]
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        current_msg = None
        matched_pattern = None
        
        # Try to find which pattern matches
        for pattern in patterns:
            for line in lines[:10]:  # Check first 10 lines
                if re.match(pattern, line):
                    matched_pattern = pattern
                    break
            if matched_pattern:
                break
        
        if not matched_pattern:
            matched_pattern = patterns[0]  # Default to first pattern
        
        for line in lines:
            match = re.match(matched_pattern, line)
            if match:
                if current_msg:
                    self.messages.append(current_msg)
                
                date_str, time_str, sender, message = match.groups()
                
                # Parse timestamp based on the matched pattern format
                try:
                    if 'AM' in time_str.upper() or 'PM' in time_str.upper():
                        # 12-hour format with seconds
                        if ':' in time_str and time_str.count(':') == 2:
                            if '/' in date_str and len(date_str.split('/')[2]) == 2:
                                timestamp = datetime.strptime(f"{date_str}, {time_str}", '%d/%m/%y, %I:%M:%S %p')
                            else:
                                timestamp = datetime.strptime(f"{date_str}, {time_str}", '%d/%m/%Y, %I:%M:%S %p')
                        # 12-hour format without seconds
                        else:
                            if '/' in date_str and len(date_str.split('/')[2]) == 2:
                                timestamp = datetime.strptime(f"{date_str}, {time_str}", '%d/%m/%y, %I:%M %p')
                            else:
                                timestamp = datetime.strptime(f"{date_str}, {time_str}", '%d/%m/%Y, %I:%M %p')
                    else:
                        # 24-hour format
                        if '/' in date_str and len(date_str.split('/')[2]) == 2:
                            timestamp = datetime.strptime(f"{date_str}, {time_str}", '%d/%m/%y, %H:%M')
                        else:
                            timestamp = datetime.strptime(f"{date_str}, {time_str}", '%d/%m/%Y, %H:%M')
                except:
                    # Fallback: try various formats
                    timestamp_str = f"{date_str}, {time_str}"
                    # Also prepare alternative versions for different separators
                    timestamp_str_at = f"{date_str} at {time_str}"
                    timestamp_str_amp = f"{date_str} @ {time_str}"
                    timestamp_str_space = f"{date_str} {time_str}"
                    for fmt in [
                        # US formats (month/day/year)
                        '%m/%d/%y, %I:%M:%S %p', '%m/%d/%Y, %I:%M:%S %p',
                        '%m/%d/%y, %I:%M %p', '%m/%d/%Y, %I:%M %p',
                        '%m/%d/%y, %H:%M:%S', '%m/%d/%Y, %H:%M:%S',
                        '%m/%d/%y, %H:%M', '%m/%d/%Y, %H:%M',
                        # European formats (day/month/year)
                        '%d/%m/%y, %I:%M:%S %p', '%d/%m/%Y, %I:%M:%S %p',
                        '%d/%m/%y, %I:%M %p', '%d/%m/%Y, %I:%M %p',
                        '%d/%m/%y, %H:%M:%S', '%d/%m/%Y, %H:%M:%S',
                        '%d/%m/%y, %H:%M', '%d/%m/%Y, %H:%M',
                        # Dash separators
                        '%d-%m-%y, %H:%M:%S', '%d-%m-%Y, %H:%M:%S',
                        '%d-%m-%y, %H:%M', '%d-%m-%Y, %H:%M',
                        '%m-%d-%y, %I:%M %p', '%m-%d-%Y, %I:%M %p',
                        # Dot separators (German/European)
                        '%d.%m.%y, %H:%M:%S', '%d.%m.%Y, %H:%M:%S',
                        '%d.%m.%y, %H:%M', '%d.%m.%Y, %H:%M',
                        # ISO format
                        '%Y-%m-%d, %H:%M:%S', '%Y-%m-%d, %H:%M',
                        # WhatsApp Business "at" format
                        '%d/%m/%y at %I:%M %p', '%d/%m/%Y at %I:%M %p',
                        '%m/%d/%y at %I:%M %p', '%m/%d/%Y at %I:%M %p',
                        # Without comma
                        '%d/%m/%y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
                        '%m/%d/%y %I:%M:%S %p', '%m/%d/%Y %I:%M:%S %p',
                        # Underscore separators
                        '%d_%m_%y, %H:%M', '%d_%m_%Y, %H:%M',
                        '%m_%d_%y, %I:%M %p', '%m_%d_%Y, %I:%M %p',
                        # yyyy/mm/dd and yyyy-mm-dd formats
                        '%Y/%m/%d, %H:%M:%S', '%Y/%m/%d, %H:%M',
                        '%Y/%m/%d, %I:%M %p', '%Y/%m/%d, %I:%M:%S %p',
                        # Using @ instead of at
                        '%d/%m/%y @ %I:%M %p', '%d/%m/%Y @ %I:%M %p',
                        '%m/%d/%y @ %I:%M %p', '%m/%d/%Y @ %I:%M %p',
                        # With milliseconds (ignore milliseconds)
                        '%d/%m/%Y, %H:%M:%S.%f', '%m/%d/%Y, %H:%M:%S.%f',
                        # Formats with timezone (ignoring timezone for now)
                        '%d/%m/%y, %I:%M %p %Z', '%d/%m/%Y, %I:%M %p %Z',
                        '%m/%d/%y, %I:%M %p %Z', '%m/%d/%Y, %I:%M %p %Z'
                    ]:
                        # Try different separator variations
                        for test_str in [timestamp_str, timestamp_str_at, timestamp_str_amp, timestamp_str_space]:
                            try:
                                timestamp = datetime.strptime(test_str, fmt)
                                break
                            except:
                                continue
                        else:
                            continue  # Continue to next format
                        break  # Break from format loop if successful
                    else:
                        continue  # Skip this message if we can't parse the date
                
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
        import re
        for topic in common_topics:
            # Use word boundary matching for short topics to avoid false positives
            if len(topic) <= 3 and ' ' not in topic:
                # For short single words, use word boundaries
                pattern = r'\b' + re.escape(topic.lower()) + r'\b'
                count = sum(1 for msg in messages if re.search(pattern, msg.lower()))
            else:
                # For longer words and phrases, use substring matching
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
        laugh_track = defaultdict(int)
        one_word_messages = defaultdict(int)
        late_night_messages = defaultdict(int)
        typo_messages = defaultdict(int)
        voice_notes = defaultdict(int)
        
        for _, row in self.df.iterrows():
            sender = row['sender']
            msg = row['message']
            hour = row['hour']
            
            emoji_count = len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿🏴-🏿]', msg))
            emoji_users[sender] += emoji_count
            
            # Laugh track - count various forms of laughter
            laugh_patterns = ['haha', 'hehe', 'lol', 'rofl', 'lmao', '😂', '🤣', '😄', '😆']
            for pattern in laugh_patterns:
                if pattern in msg.lower():
                    laugh_track[sender] += 1
                    break
            
            # One word wonder - messages with single word (excluding media)
            if not msg.startswith('<Media omitted>') and len(msg.split()) == 1 and len(msg) < 20:
                one_word_messages[sender] += 1
            
            # Late night texter (2 AM - 5 AM)
            if 2 <= hour < 5:
                late_night_messages[sender] += 1
            
            # Voice notes (look for audio omitted)
            if 'audio omitted' in msg.lower():
                voice_notes[sender] += 1
            
            # Basic typo detection (messages with multiple consecutive same letters)
            if re.search(r'(.)\1{2,}', msg) or re.search(r'[a-zA-Z]{15,}', msg):
                typo_messages[sender] += 1
            
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
        
        if laugh_track:
            print(f"🤣 Laugh Factory: {max(laugh_track, key=laugh_track.get)} (made people laugh most)")
        
        if one_word_messages:
            print(f"🔤 One Word Wonder: {max(one_word_messages, key=one_word_messages.get)}")
        
        if late_night_messages:
            print(f"🦉 Night Owl Champion: {max(late_night_messages, key=late_night_messages.get)}")
        
        if voice_notes:
            print(f"🎤 Voice Note Addict: {max(voice_notes, key=voice_notes.get)}")
        
        avg_msg_length = self.df.groupby('sender')['message'].apply(lambda x: x.str.len().mean())
        print(f"📜 Essay Writer: {avg_msg_length.idxmax()} ({avg_msg_length.max():.0f} chars avg)")
        print(f"🔤 Short & Sweet: {avg_msg_length.idxmin()} ({avg_msg_length.min():.0f} chars avg)")
        
        return {
            'emoji_champion': max(emoji_users, key=emoji_users.get) if emoji_users else None,
            'question_master': max(question_users, key=question_users.get) if question_users else None,
            'excitement_person': max(exclamation_users, key=exclamation_users.get) if exclamation_users else None,
            'laugh_factory': max(laugh_track, key=laugh_track.get) if laugh_track else None,
            'one_word_wonder': max(one_word_messages, key=one_word_messages.get) if one_word_messages else None,
            'night_owl': max(late_night_messages, key=late_night_messages.get) if late_night_messages else None
        }
    
    def generate_most_likely_awards(self):
        """Generate 'Most Likely To...' awards"""
        print("\n🎭 MOST LIKELY TO...")
        print("=" * 50)
        
        # Calculate various metrics for each user
        user_stats = {}
        for sender in self.df['sender'].unique():
            sender_msgs = self.df[self.df['sender'] == sender]
            user_stats[sender] = {
                'total_messages': len(sender_msgs),
                'avg_hour': sender_msgs['hour'].mean(),
                'weekend_ratio': len(sender_msgs[sender_msgs['day_of_week'].isin(['Saturday', 'Sunday'])]) / max(len(sender_msgs), 1),
                'night_messages': len(sender_msgs[(sender_msgs['hour'] >= 0) & (sender_msgs['hour'] < 5)]),
                'consecutive_messages': self.count_consecutive_messages(sender),
                'media_count': len(sender_msgs[sender_msgs['message'].str.contains('Media omitted')]),
                'question_ratio': len(sender_msgs[sender_msgs['message'].str.contains('?')]) / max(len(sender_msgs), 1),
                'link_count': len(sender_msgs[sender_msgs['message'].str.contains('http|www', case=False, regex=True)])
            }
        
        # Award assignments
        awards = []
        
        # Most likely to reply at 3 AM
        night_owl = max(user_stats.items(), key=lambda x: x[1]['night_messages'])
        if night_owl[1]['night_messages'] > 0:
            awards.append(f"🌙 Reply at 3 AM: {night_owl[0]} ({night_owl[1]['night_messages']} night messages)")
        
        # Most likely to send 20 messages in a row
        spam_king = max(user_stats.items(), key=lambda x: x[1]['consecutive_messages'])
        if spam_king[1]['consecutive_messages'] > 3:
            awards.append(f"📱 Send 20 messages in a row: {spam_king[0]} (max streak: {spam_king[1]['consecutive_messages']})")
        
        # Most likely to share links/news
        if any(stats['link_count'] > 0 for stats in user_stats.values()):
            news_sharer = max(user_stats.items(), key=lambda x: x[1]['link_count'])
            awards.append(f"📰 Share breaking news: {news_sharer[0]} ({news_sharer[1]['link_count']} links shared)")
        
        # Most likely to ask questions
        curious_cat = max(user_stats.items(), key=lambda x: x[1]['question_ratio'])
        if curious_cat[1]['question_ratio'] > 0.1:
            awards.append(f"❓ Ask 'but why?': {curious_cat[0]} ({curious_cat[1]['question_ratio']*100:.0f}% questions)")
        
        # Most likely to ghost the group (lurker)
        total_messages = len(self.df)
        lurkers = [(name, stats['total_messages']) for name, stats in user_stats.items() 
                   if stats['total_messages'] < total_messages * 0.02]
        if lurkers:
            biggest_lurker = min(lurkers, key=lambda x: x[1])
            awards.append(f"👻 Ghost the group: {biggest_lurker[0]} (only {biggest_lurker[1]} messages)")
        
        for award in awards:
            print(f"   • {award}")
        
        return awards
    
    def count_consecutive_messages(self, sender):
        """Count maximum consecutive messages by a sender"""
        max_consecutive = 0
        current_consecutive = 0
        
        for _, row in self.df.iterrows():
            if row['sender'] == sender:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def detect_lurkers(self):
        """Detect lurkers in the group"""
        print("\n👀 LURKER ANALYSIS")
        print("=" * 50)
        
        total_messages = len(self.df)
        total_days = (self.df['timestamp'].max() - self.df['timestamp'].min()).days
        
        lurker_scores = []
        for sender in self.df['sender'].unique():
            sender_msgs = self.df[self.df['sender'] == sender]
            message_count = len(sender_msgs)
            participation_rate = message_count / total_messages
            
            if participation_rate < 0.05:  # Less than 5% of total messages
                msgs_per_day = message_count / max(total_days, 1)
                lurker_scores.append({
                    'name': sender,
                    'messages': message_count,
                    'percentage': participation_rate * 100,
                    'msgs_per_day': msgs_per_day
                })
        
        if lurker_scores:
            lurker_scores.sort(key=lambda x: x['messages'])
            print("🤫 Silent watchers:")
            for lurker in lurker_scores[:3]:
                print(f"   • {lurker['name']}: {lurker['messages']} messages ({lurker['percentage']:.1f}% participation)")
                if lurker['msgs_per_day'] < 0.1:
                    print(f"     → Appears once every {1/max(lurker['msgs_per_day'], 0.001):.0f} days")
        else:
            print("   Everyone actively participates! No lurkers detected.")
        
        return lurker_scores
    
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
    
    most_likely = analyzer.generate_most_likely_awards()
    
    lurkers = analyzer.detect_lurkers()
    
    fig = analyzer.create_visualizations()
    
    print("\n" + "=" * 60)
    print("🎭 Analysis Complete! Share these fun insights with your group!")
    print("=" * 60)

if __name__ == "__main__":
    main()