import streamlit as st
import pandas as pd
import tempfile
import os
import zipfile
from whatsapp_analyzer import WhatsAppAnalyzer
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(
    page_title="WhatsApp Group Analyzer",
    page_icon="💬",
    layout="wide"
)

def main():
    st.title("💬 WhatsApp Group Chat Analyzer")
    st.markdown("### Upload your WhatsApp chat export and get fun insights about your group!")
    
    st.markdown("""
    **How to export your WhatsApp chat:**
    1. Open WhatsApp group chat
    2. Tap menu (⋮) → More → Export chat
    3. Choose 'Without media' 
    4. Upload the .zip file here (we'll extract the chat automatically!)
    """)
    
    uploaded_file = st.file_uploader(
        "Choose your WhatsApp chat file", 
        type=['zip', 'txt'],
        help="Upload the .zip file exported from WhatsApp (or .txt if you already extracted it)"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing your chat... This might take a moment!"):
            try:
                # Handle ZIP or TXT files
                if uploaded_file.name.endswith('.zip'):
                    # Extract ZIP file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                        tmp_zip.write(uploaded_file.getvalue())
                        tmp_zip_path = tmp_zip.name
                    
                    # Extract and find _chat.txt
                    with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                        file_list = zip_ref.namelist()
                        chat_file = None
                        
                        # Look for _chat.txt file
                        for file in file_list:
                            if file.endswith('_chat.txt') or file.endswith('.txt'):
                                chat_file = file
                                break
                        
                        if not chat_file:
                            st.error("❌ No chat file found in ZIP. Please make sure you exported the chat correctly.")
                            return
                        
                        # Extract the chat file
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
                            content = zip_ref.read(chat_file).decode('utf-8', errors='ignore')
                            tmp_file.write(content)
                            tmp_file_path = tmp_file.name
                    
                    os.unlink(tmp_zip_path)
                else:
                    # Handle direct TXT file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
                        content = uploaded_file.getvalue().decode('utf-8')
                        tmp_file.write(content)
                        tmp_file_path = tmp_file.name
                
                analyzer = WhatsAppAnalyzer(tmp_file_path)
                df = analyzer.parse_chat()
                
                if len(df) == 0:
                    st.error("❌ No messages found! Please check your file format.")
                    return
                
                st.success(f"✅ Successfully parsed {len(df)} messages from {len(df['sender'].unique())} participants!")
                
                # Generate shareable report
                share_text = generate_share_report(analyzer)
                
                # Prominent share section
                st.markdown("---")
                st.markdown("""
                <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <h3>📤 Share Your Group Analysis!</h3>
                <p>Share the fun insights with your group members!</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if st.button("📱 Share on WhatsApp", type="primary", use_container_width=True):
                        import urllib.parse
                        encoded_text = urllib.parse.quote(share_text[:1000])  # WhatsApp has character limits
                        whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_text}"
                        
                        # Mobile-friendly sharing
                        st.markdown(f"""
                        <script>
                        if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {{
                            window.open('{whatsapp_url}', '_blank');
                        }}
                        </script>
                        """, unsafe_allow_html=True)
                        
                        st.success("📱 WhatsApp share link generated!")
                        st.markdown(f'**[👆 Click here to share on WhatsApp]({whatsapp_url})**')
                        st.info("💡 On mobile: Link opens WhatsApp directly! On desktop: Opens WhatsApp Web.")
                
                with col2:
                    if st.button("📋 Copy Report", use_container_width=True):
                        st.code(share_text, language="text")
                        st.success("📋 Report copied! Use Ctrl+A, Ctrl+C to copy the text above.")
                
                with col3:
                    with st.popover("📄 Preview Report"):
                        st.text_area("Report Preview", share_text, height=300)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Messages", f"{len(df):,}")
                    st.metric("Participants", len(df['sender'].unique()))
                
                with col2:
                    total_days = (df['timestamp'].max() - df['timestamp'].min()).days
                    msgs_per_day = len(df) / max(total_days, 1)
                    st.metric("Days Analyzed", total_days)
                    st.metric("Msgs/Day", f"{msgs_per_day:.1f}")
                
                tabs = st.tabs(["📊 Overview", "🏆 Top Chatters", "😊 Sentiment", "🔥 Hot Topics", "⚡ Drama Topics", "⏰ Time Patterns", "🎭 Fun Awards"])
                
                with tabs[0]:
                    st.header("📊 Chat Overview")
                    display_overview(df, analyzer)
                
                with tabs[1]:
                    st.header("🏆 Most Active Members")
                    display_top_chatters(analyzer)
                
                with tabs[2]:
                    st.header("😊 Sentiment Analysis")
                    display_sentiment(analyzer)
                
                with tabs[3]:
                    st.header("🔥 Hot Topics")
                    display_hot_topics(analyzer)
                
                with tabs[4]:
                    st.header("⚡ Most Polarizing Topics")
                    display_polarizing_topics(analyzer)
                
                with tabs[5]:
                    st.header("⏰ Activity Patterns")
                    display_time_patterns(analyzer)
                
                with tabs[6]:
                    st.header("🎭 Fun Awards")
                    display_fun_awards(analyzer)
                
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"❌ Error analyzing chat: {str(e)}")
                st.info("""
                **Troubleshooting:**
                - Make sure you uploaded a valid WhatsApp chat export ZIP file
                - The ZIP should contain a file ending with '_chat.txt'
                - Try exporting the chat again from WhatsApp: Group → Menu (⋮) → More → Export chat → Without media
                """)
                import traceback
                with st.expander("Technical Details (for debugging)"):
                    st.code(traceback.format_exc())

def display_overview(df, analyzer):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Activity Over Time")
        daily_msgs = df.groupby('date').size()
        st.line_chart(daily_msgs)
    
    with col2:
        st.subheader("📅 Messages by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_activity = df.groupby('day_of_week').size().reindex(day_order)
        st.bar_chart(daily_activity)

def display_top_chatters(analyzer):
    activity_stats = []
    for sender in analyzer.df['sender'].unique():
        sender_msgs = analyzer.df[analyzer.df['sender'] == sender]
        total_msgs = len(sender_msgs)
        avg_length = sender_msgs['message'].str.len().mean()
        
        activity_stats.append({
            'Name': sender,
            'Total Messages': total_msgs,
            'Avg Message Length': f"{avg_length:.0f} chars",
            'Peak Hour': f"{sender_msgs.groupby('hour').size().idxmax()}:00",
            'Favorite Day': sender_msgs.groupby('day_of_week').size().idxmax()
        })
    
    activity_df = pd.DataFrame(activity_stats).sort_values('Total Messages', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥇 Top 10 Chatters")
        top_10 = activity_df.head(10)
        st.dataframe(top_10, use_container_width=True)
    
    with col2:
        st.subheader("📊 Message Distribution")
        top_senders = analyzer.df['sender'].value_counts().head(10)
        st.bar_chart(top_senders)

def display_sentiment(analyzer):
    sentiments = []
    for _, row in analyzer.df.iterrows():
        if not row['message'].startswith('<Media omitted>'):
            from textblob import TextBlob
            blob = TextBlob(row['message'])
            sentiments.append({
                'sender': row['sender'],
                'polarity': blob.sentiment.polarity
            })
    
    if sentiments:
        sentiment_df = pd.DataFrame(sentiments)
        sender_sentiment = sentiment_df.groupby('sender')['polarity'].mean().round(3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("😄 Most Positive People")
            top_positive = sender_sentiment.nlargest(5)
            for sender, score in top_positive.items():
                emoji = '😄' if score > 0.3 else '🙂' if score > 0 else '😐'
                st.write(f"{emoji} **{sender}**: {score:.3f}")
        
        with col2:
            st.subheader("😔 Most Negative/Neutral")
            top_negative = sender_sentiment.nsmallest(5)
            for sender, score in top_negative.items():
                emoji = '😔' if score < -0.1 else '😐'
                st.write(f"{emoji} **{sender}**: {score:.3f}")

def display_hot_topics(analyzer):
    messages = [msg for msg in analyzer.df['message'] if not msg.startswith('<Media omitted>')]
    
    # Comprehensive Indian topics
    common_topics = [
        # Politics & Government
        'modi', 'bjp', 'congress', 'aap', 'election', 'vote', 'parliament', 'politics', 'government', 
        'rahul gandhi', 'amit shah', 'kejriwal', 'mamata', 'yogi', 'state election', 'lok sabha', 'rajya sabha',
        
        # Finance & Economy
        'stock', 'market', 'nifty', 'sensex', 'investment', 'mutual fund', 'sip', 'fd', 'ppf', 'nsc',
        'crypto', 'bitcoin', 'trading', 'portfolio', 'gold', 'silver', 'inflation', 'gst', 'budget',
        'rupee', 'dollar', 'bank', 'loan', 'emi', 'insurance', 'tax', 'income tax', 'epf', 'pf',
        
        # International
        'usa', 'america', 'trump', 'biden', 'china', 'pakistan', 'ukraine', 'russia', 'israel',
        'middle east', 'europe', 'japan', 'singapore', 'dubai', 'saudi', 'immigration', 'visa',
        
        # Cities & States
        'bangalore', 'bengaluru', 'mumbai', 'delhi', 'chennai', 'hyderabad', 'pune', 'kolkata',
        'gurgaon', 'noida', 'karnataka', 'maharashtra', 'tamil nadu', 'kerala', 'gujarat', 'rajasthan',
        'up', 'bihar', 'west bengal', 'telangana', 'andhra pradesh',
        
        # Technology & Work
        'work', 'job', 'salary', 'appraisal', 'promotion', 'interview', 'tech', 'software', 'ai',
        'google', 'microsoft', 'amazon', 'infosys', 'tcs', 'wipro', 'startup', 'coding', 'python',
        'java', 'cloud', 'aws', 'azure', 'meeting', 'office', 'wfh', 'remote', 'onsite',
        
        # Entertainment & Sports
        'movie', 'bollywood', 'hollywood', 'netflix', 'amazon prime', 'hotstar', 'film', 'actor',
        'cricket', 'ipl', 'world cup', 'kohli', 'dhoni', 'rohit', 'football', 'fifa', 'olympics',
        'tennis', 'badminton', 'kabaddi', 'hockey', 'sports',
        
        # Food & Culture
        'food', 'biryani', 'dosa', 'samosa', 'chai', 'coffee', 'restaurant', 'zomato', 'swiggy',
        'lunch', 'dinner', 'breakfast', 'street food', 'festival', 'diwali', 'holi', 'eid',
        'ganesh chaturthi', 'durga puja', 'navratri', 'dussehra', 'karva chauth', 'wedding',
        
        # Transportation & Travel
        'uber', 'ola', 'auto', 'metro', 'train', 'flight', 'airport', 'traffic', 'petrol',
        'diesel', 'car', 'bike', 'travel', 'vacation', 'holiday', 'goa', 'kerala', 'kashmir',
        'himachal', 'uttarakhand', 'rajasthan', 'abroad',
        
        # Daily Life
        'house', 'rent', 'flat', 'apartment', 'pg', 'roommate', 'electricity', 'water', 'wifi',
        'shopping', 'flipkart', 'amazon', 'myntra', 'grocery', 'vegetables', 'market',
        'hospital', 'doctor', 'medicine', 'health', 'gym', 'fitness',
        
        # Social & Personal
        'family', 'parents', 'marriage', 'relationship', 'friends', 'party', 'weekend', 'birthday',
        'anniversary', 'plan', 'tomorrow', 'tonight', 'meeting', 'college', 'school', 'education',
        'exam', 'results', 'admission',
        
        # Weather & Seasons
        'weather', 'rain', 'monsoon', 'summer', 'winter', 'heat', 'cold', 'humidity', 'climate',
        
        # Religion & Spirituality
        'temple', 'church', 'mosque', 'gurudwara', 'god', 'prayer', 'bhajan', 'mandir', 'devotion',
        'spiritual', 'meditation', 'yoga',
        
        # News & Current Events
        'news', 'breaking news', 'update', 'corona', 'covid', 'vaccine', 'lockdown', 'pandemic',
        'economy', 'recession', 'growth', 'development'
    ]
    
    topic_counts = {}
    for topic in common_topics:
        count = sum(1 for msg in messages if topic.lower() in msg.lower())
        if count > 0:
            topic_counts[topic.capitalize()] = count
    
    if topic_counts:
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗣️ Most Discussed Topics")
            for i, (topic, count) in enumerate(sorted_topics[:10]):
                st.write(f"{i+1}. **{topic}**: {count} mentions")
        
        with col2:
            st.subheader("📊 Topic Frequency")
            topic_df = pd.DataFrame(sorted_topics[:8], columns=['Topic', 'Count'])
            st.bar_chart(topic_df.set_index('Topic'))

def display_polarizing_topics(analyzer):
    from collections import defaultdict
    import numpy as np
    from textblob import TextBlob
    
    topic_sentiments = defaultdict(list)
    topics_to_check = [
        # High polarization topics
        'trump', 'biden', 'usa', 'china', 'pakistan', 'election', 'politics', 'modi', 'bjp', 'congress',
        'crypto', 'bitcoin', 'stock', 'market', 'investment', 'gold', 'economy', 'recession',
        'movie', 'bollywood', 'cricket', 'kohli', 'dhoni', 'ipl', 'football',
        'work', 'salary', 'job', 'office', 'wfh', 'onsite', 
        'bangalore', 'mumbai', 'delhi', 'traffic', 'rent', 'house',
        'food', 'restaurant', 'biryani', 'weekend', 'party', 'travel', 'vacation',
        'weather', 'rain', 'monsoon', 'summer', 'covid', 'vaccine', 'lockdown'
    ]
    
    for _, row in analyzer.df.iterrows():
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
    
    if polarizing:
        polarizing.sort(key=lambda x: x[1], reverse=True)
        
        st.subheader("🔄 Topics That Create Most Drama")
        st.write("*Higher variance = more polarizing opinions*")
        
        for i, (topic, std, count) in enumerate(polarizing[:8]):
            drama_level = "🔥🔥🔥" if std > 0.3 else "🔥🔥" if std > 0.2 else "🔥"
            st.write(f"{drama_level} **{topic.capitalize()}**: σ={std:.3f} ({count} mentions)")

def display_time_patterns(analyzer):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Hourly Activity")
        hourly = analyzer.df.groupby('hour').size()
        st.line_chart(hourly)
        
        peak_hour = hourly.idxmax()
        st.write(f"🌟 **Peak hour**: {peak_hour}:00 ({hourly.max()} messages)")
    
    with col2:
        st.subheader("📅 Daily Activity")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = analyzer.df.groupby('day_of_week').size().reindex(day_order)
        st.bar_chart(daily)
        
        peak_day = daily.idxmax()
        st.write(f"🗓️ **Most active day**: {peak_day} ({daily.max()} messages)")

def display_fun_awards(analyzer):
    from collections import defaultdict
    import re
    
    emoji_users = defaultdict(int)
    question_users = defaultdict(int)
    exclamation_users = defaultdict(int)
    
    for _, row in analyzer.df.iterrows():
        sender = row['sender']
        msg = row['message']
        
        emoji_count = len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿🏴-🏿]', msg))
        emoji_users[sender] += emoji_count
        
        if '?' in msg:
            question_users[sender] += 1
        if '!' in msg:
            exclamation_users[sender] += 1
    
    avg_msg_length = analyzer.df.groupby('sender')['message'].apply(lambda x: x.str.len().mean())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Communication Awards")
        if emoji_users:
            st.write(f"😂 **Emoji Champion**: {max(emoji_users, key=emoji_users.get)}")
        if question_users:
            st.write(f"❓ **Question Master**: {max(question_users, key=question_users.get)}")
        if exclamation_users:
            st.write(f"❗ **Excitement Champion**: {max(exclamation_users, key=exclamation_users.get)}")
    
    with col2:
        st.subheader("📝 Writing Style Awards")
        st.write(f"📜 **Essay Writer**: {avg_msg_length.idxmax()} ({avg_msg_length.max():.0f} chars avg)")
        st.write(f"🔤 **Short & Sweet**: {avg_msg_length.idxmin()} ({avg_msg_length.min():.0f} chars avg)")
    
    st.subheader("🚀 Conversation Starters")
    analyzer.df['time_diff'] = analyzer.df['timestamp'].diff()
    conversation_starters = []
    
    for idx, row in analyzer.df.iterrows():
        if idx == 0 or row['time_diff'].total_seconds() > 3600:
            conversation_starters.append(row['sender'])
    
    from collections import Counter
    starter_counts = Counter(conversation_starters)
    
    for i, (sender, count) in enumerate(starter_counts.most_common(5)):
        st.write(f"{i+1}. **{sender}**: started {count} conversations")

def generate_share_report(analyzer):
    """Generate a shareable text report"""
    df = analyzer.df
    
    # Basic stats
    total_days = (df['timestamp'].max() - df['timestamp'].min()).days
    msgs_per_day = len(df) / max(total_days, 1)
    
    # Get top chatters
    activity_stats = []
    for sender in df['sender'].unique():
        sender_msgs = df[df['sender'] == sender]
        activity_stats.append({
            'name': sender,
            'total_messages': len(sender_msgs),
            'avg_message_length': sender_msgs['message'].str.len().mean(),
            'most_active_hour': sender_msgs.groupby('hour').size().idxmax(),
            'favorite_day': sender_msgs.groupby('day_of_week').size().idxmax()
        })
    
    activity_df = pd.DataFrame(activity_stats).sort_values('total_messages', ascending=False)
    top_5 = activity_df.head(5)
    
    # Get sentiment analysis
    from textblob import TextBlob
    sentiments = []
    for _, row in df.iterrows():
        if not row['message'].startswith('<Media omitted>'):
            blob = TextBlob(row['message'])
            sentiments.append({
                'sender': row['sender'],
                'polarity': blob.sentiment.polarity
            })
    
    if sentiments:
        sentiment_df = pd.DataFrame(sentiments)
        sender_sentiment = sentiment_df.groupby('sender')['polarity'].mean().round(3)
        most_positive = sender_sentiment.nlargest(3)
    
    # Get hot topics
    messages = [msg for msg in df['message'] if not msg.startswith('<Media omitted>')]
    
    common_topics = [
        'modi', 'bjp', 'congress', 'trump', 'usa', 'china', 'pakistan', 'election', 'politics',
        'stock', 'crypto', 'bitcoin', 'gold', 'investment', 'nifty', 'sensex', 'market',
        'bangalore', 'mumbai', 'delhi', 'chennai', 'hyderabad', 'work', 'job', 'salary',
        'movie', 'bollywood', 'cricket', 'ipl', 'kohli', 'food', 'biryani', 'restaurant',
        'travel', 'vacation', 'house', 'rent', 'traffic', 'weather', 'rain', 'monsoon',
        'festival', 'diwali', 'holi', 'wedding', 'family', 'friends', 'party', 'weekend'
    ]
    
    topic_counts = {}
    for topic in common_topics:
        count = sum(1 for msg in messages if topic.lower() in msg.lower())
        if count > 0:
            topic_counts[topic.capitalize()] = count
    
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Time patterns
    hourly = df.groupby('hour').size()
    daily = df.groupby('day_of_week').size()
    peak_hour = hourly.idxmax()
    peak_day = daily.idxmax()
    
    # Get awards
    from collections import defaultdict
    import re
    
    emoji_users = defaultdict(int)
    question_users = defaultdict(int)
    exclamation_users = defaultdict(int)
    
    for _, row in df.iterrows():
        sender = row['sender']
        msg = row['message']
        
        emoji_count = len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿🏴-🏿]', msg))
        emoji_users[sender] += emoji_count
        
        if '?' in msg:
            question_users[sender] += 1
        if '!' in msg:
            exclamation_users[sender] += 1
    
    avg_msg_length = df.groupby('sender')['message'].apply(lambda x: x.str.len().mean())
    
    # Generate the report text
    report = f"""🎉 WHATSAPP GROUP ANALYSIS REPORT 🎉

📈 *Quick Stats*
• Total messages: {len(df):,}
• Active members: {len(df['sender'].unique())}
• Time period: {total_days} days
• Average: {msgs_per_day:.1f} messages/day

🏆 *Top 5 Chatters*
"""
    
    for i, (_, row) in enumerate(top_5.iterrows()):
        medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        medal = medals[i] if i < 5 else f"{i+1}."
        report += f"{medal} {row['name']}: {row['total_messages']} messages\n"
    
    if sentiments:
        report += f"""
😊 *Mood Analysis*
Most Positive Vibes:
"""
        for sender, score in most_positive.head(3).items():
            emoji = '😄' if score > 0.3 else '🙂'
            report += f"{emoji} {sender}: {score:.3f}\n"
    
    report += f"""
🔥 *Hot Topics*
"""
    for topic, count in sorted_topics:
        report += f"• {topic}: {count} mentions\n"
    
    report += f"""
⏰ *Group Behavior*
• Peak hour: {peak_hour}:00
• Most active day: {peak_day}
• Communication style: {'Early birds!' if peak_hour < 10 else 'Night owls!' if peak_hour > 20 else 'Steady chatters!'}

🏅 *Fun Awards*
"""
    
    if emoji_users:
        report += f"😂 Emoji Champion: {max(emoji_users, key=emoji_users.get)}\n"
    if question_users:
        report += f"❓ Question Master: {max(question_users, key=question_users.get)}\n"
    if exclamation_users:
        report += f"❗ Excitement Champion: {max(exclamation_users, key=exclamation_users.get)}\n"
    
    report += f"📜 Essay Writer: {avg_msg_length.idxmax()} ({avg_msg_length.max():.0f} chars avg)\n"
    report += f"🔤 Short & Sweet: {avg_msg_length.idxmin()} ({avg_msg_length.min():.0f} chars avg)\n"
    
    report += f"""
---
🚀 *Want to analyze YOUR group chat?*

Try the WhatsApp Group Analyzer for FREE!
Just upload your chat export and discover:
• Who are the real conversation starters
• What topics create the most drama  
• Your group's personality insights
• Peak activity times & patterns

Analyze your chat now! 📊✨
"""
    
    return report

if __name__ == "__main__":
    main()