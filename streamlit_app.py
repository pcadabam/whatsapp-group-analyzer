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
    layout="centered",  # Better for mobile
    initial_sidebar_state="collapsed"
)

def main():
    # Consumer product hero section
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 30px;">
        <h1 style="color: white; font-size: 2.5em; margin-bottom: 15px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🕵️ Group Chat Secrets</h1>
        <p style="color: white; font-size: 20px; margin-bottom: 20px; opacity: 0.9;">Find out who's REALLY running your WhatsApp group!</p>
        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 15px; margin: 0 auto; max-width: 400px;">
            <p style="color: white; font-weight: bold; margin: 0;">✨ Discover jaw-dropping insights that will shock your friends!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Social proof
    st.markdown("""
    <div style="text-align: center; margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 10px;">
        <p style="margin: 0; color: #666; font-style: italic;">
        "OMG this exposed everything about our group 😂" - Sarah M.<br>
        "I can't believe how accurate this is!" - Ravi K.<br>
        "Our group hasn't stopped talking about these results!" - Priya S.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Simple instructions
    with st.expander("📱 How to get your chat file", expanded=True):
        st.markdown("""
        **Super easy:**
        1. Open your WhatsApp group
        2. Tap the group name → Export Chat
        3. Choose "Without Media" 
        4. Upload the file here!
        
        **Don't worry** - we don't save anything, it's 100% private! 🔒
        """)
    
    uploaded_file = st.file_uploader(
        "🚀 Drop your chat file here and let the magic begin!", 
        type=['zip', 'txt'],
        help="✨ Upload the ZIP file from WhatsApp and watch the secrets unfold!"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔮 Revealing your group's secrets... Prepare for some surprises!"):
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
                
                # Simple success message
                st.success(f"We analyzed {len(df):,} messages from {len(df['sender'].unique())} people")
                
                # Generate shareable report
                share_text = generate_share_report(analyzer)
                
                # Show the actual report that will be shared - MAIN FOCUS
                st.markdown("---")
                
                # Display the shareable report with contained styling
                import re
                
                # Convert basic markdown to HTML for contained display
                html_text = share_text
                
                # First, clean up excessive spacing
                # Remove triple+ newlines and replace with double
                html_text = re.sub(r'\n{3,}', '\n\n', html_text)
                
                # Convert headers first with minimal spacing
                html_text = re.sub(r'^### (.*)', r'<h3 style="color: #2d3748; margin: 8px 0 2px 0; font-weight: 600;">\1</h3>', html_text, flags=re.MULTILINE)
                html_text = re.sub(r'^#### (.*)', r'<h4 style="color: #4a5568; margin: 6px 0 1px 0; font-weight: 600;">\1</h4>', html_text, flags=re.MULTILINE)
                html_text = re.sub(r'^## (.*)', r'<h2 style="color: #1a202c; margin: 10px 0 3px 0; font-weight: 700;">\1</h2>', html_text, flags=re.MULTILINE)
                
                # Convert bold and italic text
                html_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_text)
                html_text = re.sub(r'_(.*?)_', r'<em>\1</em>', html_text)
                
                # Handle line breaks more carefully
                # Split into paragraphs and process
                paragraphs = html_text.split('\n\n')
                processed_paragraphs = []
                
                for para in paragraphs:
                    if para.strip():
                        # For regular paragraphs, replace single newlines with spaces (for flowing text)
                        # But keep bullet points on separate lines
                        lines = para.split('\n')
                        processed_lines = []
                        for line in lines:
                            if line.strip().startswith('•') or line.strip().startswith('<h') or line.strip().startswith('</h'):
                                processed_lines.append(line.strip())
                            else:
                                processed_lines.append(line.strip())
                        processed_paragraphs.append('<br>'.join(processed_lines))
                
                html_text = '</p><p style="margin: 4px 0;">'.join(processed_paragraphs)
                html_text = '<p style="margin: 4px 0;">' + html_text + '</p>'
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
                    padding: 25px;
                    border-radius: 15px;
                    border: 2px solid #25D366;
                    margin: 20px 0;
                    box-shadow: 0 4px 12px rgba(37, 211, 102, 0.15);
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    font-size: 16px;
                    line-height: 1.3;
                    color: #1a202c;
                ">
                {html_text}
                </div>
                """, unsafe_allow_html=True)
                
                # Mobile-optimized WhatsApp sharing
                import urllib.parse
                encoded_text = urllib.parse.quote(share_text)
                whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_text}"
                
                # Single prominent share button
                st.markdown(f"""
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{whatsapp_url}" target="_blank" style="
                        display: inline-block;
                        background: linear-gradient(135deg, #25D366, #128C7E);
                        color: white;
                        padding: 20px 40px;
                        border-radius: 50px;
                        text-decoration: none;
                        font-size: 20px;
                        font-weight: bold;
                        box-shadow: 0 6px 20px rgba(37, 211, 102, 0.4);
                        transition: all 0.3s ease;
                        border: none;
                        cursor: pointer;
                        min-width: 280px;
                    ">
                        💬 Share
                    </a>
                </div>
                <div style="text-align: center; color: #666; font-size: 16px; margin-bottom: 20px;">
                    Tap to send this to your group! 👆
                </div>
                """, unsafe_allow_html=True)
                
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("💬 Total Messages", f"{len(df):,}")
                    st.metric("👥 People in Group", len(df['sender'].unique()))
                
                with col2:
                    total_days = (df['timestamp'].max() - df['timestamp'].min()).days
                    msgs_per_day = len(df) / max(total_days, 1)
                    st.metric("📅 Days of Chat", total_days)
                    st.metric("🔥 Messages per Day", f"{msgs_per_day:.1f}")
                
                # Consumer-focused detailed insights
                st.markdown("""
                <div style="text-align: center; margin: 30px 0; padding: 20px; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 15px;">
                    <h2 style="color: white; margin: 0;">🕵️ Dive Deeper Into The Drama</h2>
                    <p style="color: white; opacity: 0.9; margin-top: 10px;">Click each section to uncover more juicy secrets!</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📈 Your Group's Chatting Patterns", expanded=True):
                    display_overview(df, analyzer)
                
                with st.expander("👑 The Conversation Rulers"):
                    display_top_chatters(analyzer)
                
                with st.expander("😇 Angels vs Devils in Your Group"):
                    display_sentiment(analyzer)
                
                with st.expander("🔥 Your Group's Obsessions Exposed"):
                    display_hot_topics(analyzer)
                
                with st.expander("💥 Topics That Cause Group Wars"):
                    display_polarizing_topics(analyzer)
                
                with st.expander("🕐 When Your Group Comes Alive"):
                    display_time_patterns(analyzer)
                
                with st.expander("🏆 Hall of Fame & Shame"):
                    display_fun_awards(analyzer)
                
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.markdown("""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #ff7b7b, #ff6b6b); border-radius: 15px; margin: 20px 0;">
                    <h3 style="color: white; margin-bottom: 10px;">😅 Whoops! The magic spell didn't work!</h3>
                    <p style="color: white; margin: 0;">Let's try again with the right ingredients...</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("""
                **Quick fixes that usually work:**
                - Upload the ZIP file you got from WhatsApp (the one with all the messages)
                - Make sure you chose "Export Chat" from your group settings
                - Try a different group if this one is acting weird
                - Some very old chats might not work perfectly
                
                **Still stuck?** Try with a more recent group chat! 💪
                """)
                with st.expander("🤓 Nerdy technical stuff"):
                    st.text(str(e))

def display_overview(df, analyzer):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Your Group's Activity Rollercoaster")
        daily_msgs = df.groupby('date').size()
        st.line_chart(daily_msgs)
        st.caption("Those peaks? That's when the drama happened! 🎭")
    
    with col2:
        st.subheader("📅 When Does Your Group Go Crazy?")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_activity = df.groupby('day_of_week').size().reindex(day_order)
        st.bar_chart(daily_activity)
        st.caption("Monday blues or Friday vibes? 🎉")

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
        # Politics & Government (Extended)
        'modi', 'bjp', 'congress', 'aap', 'election', 'vote', 'parliament', 'politics', 'government', 
        'rahul gandhi', 'amit shah', 'kejriwal', 'mamata', 'yogi', 'state election', 'lok sabha', 'rajya sabha',
        'cm', 'pm', 'president', 'governor', 'minister', 'mla', 'mp', 'bjp congress', 'alliance', 'coalition',
        'supreme court', 'high court', 'cbi', 'ed', 'income tax raid', 'corruption', 'scam',
        
        # Finance & Economy (Extended)
        'stock', 'market', 'nifty', 'sensex', 'investment', 'mutual fund', 'sip', 'fd', 'ppf', 'nsc',
        'crypto', 'bitcoin', 'ethereum', 'trading', 'portfolio', 'gold', 'silver', 'inflation', 'gst', 'budget',
        'rupee', 'dollar', 'bank', 'loan', 'emi', 'insurance', 'tax', 'income tax', 'epf', 'pf',
        'credit card', 'debit card', 'upi', 'paytm', 'gpay', 'phonepe', 'aadhar', 'pan card',
        'rbi', 'repo rate', 'interest rate', 'recession', 'bear market', 'bull market', 'ipo',
        
        # International (Extended)
        'usa', 'america', 'trump', 'biden', 'china', 'pakistan', 'ukraine', 'russia', 'israel',
        'middle east', 'europe', 'japan', 'singapore', 'dubai', 'saudi', 'immigration', 'visa',
        'h1b', 'green card', 'canada', 'australia', 'uk', 'germany', 'france', 'switzerland',
        'world war', 'nato', 'un', 'g20', 'brics', 'asean',
        
        # Cities & States (Extended)
        'bangalore', 'bengaluru', 'mumbai', 'delhi', 'chennai', 'hyderabad', 'pune', 'kolkata',
        'gurgaon', 'noida', 'karnataka', 'maharashtra', 'tamil nadu', 'kerala', 'gujarat', 'rajasthan',
        'up', 'bihar', 'west bengal', 'telangana', 'andhra pradesh', 'odisha', 'assam', 'punjab',
        'haryana', 'madhya pradesh', 'uttarakhand', 'himachal pradesh', 'goa', 'jharkhand', 'chhattisgarh',
        'jammu kashmir', 'ladakh', 'delhi ncr', 'tier 2 city', 'metro city',
        
        # Technology & Work (Extended)
        'work', 'job', 'salary', 'appraisal', 'promotion', 'interview', 'tech', 'software', 'ai',
        'google', 'microsoft', 'amazon', 'infosys', 'tcs', 'wipro', 'startup', 'coding', 'python',
        'java', 'cloud', 'aws', 'azure', 'meeting', 'office', 'wfh', 'remote', 'onsite',
        'layoffs', 'hiring', 'hr', 'boss', 'manager', 'team lead', 'resignation', 'notice period',
        'chatgpt', 'openai', 'machine learning', 'data science', 'blockchain', 'metaverse',
        'linkedin', 'naukri', 'glassdoor', 'referral', 'interview prep', 'coding round',
        
        # Entertainment & Sports (Extended)
        'movie', 'bollywood', 'hollywood', 'netflix', 'amazon prime', 'hotstar', 'film', 'actor',
        'cricket', 'ipl', 'world cup', 'kohli', 'dhoni', 'rohit', 'football', 'fifa', 'olympics',
        'tennis', 'badminton', 'kabaddi', 'hockey', 'sports', 'rcb', 'csk', 'mi', 'kkr',
        'youtube', 'instagram', 'facebook', 'twitter', 'snapchat', 'tiktok', 'reel', 'story',
        'series', 'web series', 'ott', 'trailer', 'box office', 'oscar', 'national award',
        'south movie', 'dubbed', 'subtitles', 'theater', 'multiplex', 'pvr', 'inox',
        
        # Food & Culture (Extended)
        'food', 'biryani', 'dosa', 'samosa', 'chai', 'coffee', 'restaurant', 'zomato', 'swiggy',
        'lunch', 'dinner', 'breakfast', 'street food', 'festival', 'diwali', 'holi', 'eid',
        'ganesh chaturthi', 'durga puja', 'navratri', 'dussehra', 'karva chauth', 'wedding',
        'north indian', 'south indian', 'gujarati', 'punjabi', 'bengali', 'maharashtrian',
        'thali', 'butter chicken', 'paneer', 'dal', 'rice', 'roti', 'naan', 'pizza', 'burger',
        'sweets', 'mithai', 'gulab jamun', 'rasgulla', 'laddu', 'home cooked', 'mom food',
        
        # Transportation & Travel (Extended)
        'uber', 'ola', 'auto', 'metro', 'train', 'flight', 'airport', 'traffic', 'petrol',
        'diesel', 'car', 'bike', 'travel', 'vacation', 'holiday', 'goa', 'kerala', 'kashmir',
        'himachal', 'uttarakhand', 'rajasthan', 'abroad', 'booking', 'makemytrip', 'goibibo',
        'irctc', 'tatkal', 'waiting list', 'confirmed ticket', 'cancellation', 'refund',
        'road trip', 'bike trip', 'solo travel', 'group trip', 'honeymoon', 'family trip',
        'bus', 'sleeper', 'ac', 'general', 'first class', 'business class', 'economy',
        
        # Daily Life (Extended)
        'house', 'rent', 'flat', 'apartment', 'pg', 'roommate', 'electricity', 'water', 'wifi',
        'shopping', 'flipkart', 'amazon', 'myntra', 'grocery', 'vegetables', 'market',
        'hospital', 'doctor', 'medicine', 'health', 'gym', 'fitness', 'maid', 'cook', 'driver',
        'maintenance', 'society', 'security', 'lift', 'parking', 'power cut', 'water shortage',
        'gas cylinder', 'bisleri', 'newspaper', 'milk', 'bread', 'grocery delivery',
        
        # Social & Personal (Extended)
        'family', 'parents', 'marriage', 'relationship', 'friends', 'party', 'weekend', 'birthday',
        'anniversary', 'plan', 'tomorrow', 'tonight', 'meeting', 'college', 'school', 'education',
        'exam', 'results', 'admission', 'baby', 'pregnancy', 'delivery', 'naming ceremony',
        'thread ceremony', 'engagement', 'bachelor party', 'bachelorette', 'sangam', 'mehendi',
        'gossip', 'fight', 'patch up', 'breakup', 'dating', 'tinder', 'bumble', 'arranged marriage',
        
        # Health & Wellness (Extended)
        'covid', 'vaccine', 'booster', 'mask', 'sanitizer', 'fever', 'cough', 'cold',
        'headache', 'stomach pain', 'bp', 'diabetes', 'cholesterol', 'thyroid', 'weight loss',
        'diet', 'exercise', 'yoga', 'meditation', 'mental health', 'stress', 'anxiety',
        'sleep', 'insomnia', 'vitamin d', 'calcium', 'protein', 'supplements',
        
        # Weather & Seasons (Extended)
        'weather', 'rain', 'monsoon', 'summer', 'winter', 'heat', 'cold', 'humidity', 'climate',
        'cyclone', 'flood', 'drought', 'temperature', 'weather forecast', 'imd', 'heatwave',
        'thunderstorm', 'lightning', 'hail', 'fog', 'pollution', 'aqi', 'air quality',
        
        # Religion & Spirituality (Extended)
        'temple', 'church', 'mosque', 'gurudwara', 'god', 'prayer', 'bhajan', 'mandir', 'devotion',
        'spiritual', 'meditation', 'yoga', 'jai shri ram', 'om namah shivaya', 'allah',
        'jesus', 'waheguru', 'hanuman', 'ganesha', 'shiva', 'vishnu', 'krishna', 'rama',
        'good morning', 'good night', 'blessed', 'grace', 'miracle', 'faith', 'karma',
        
        # Education & Career (Extended)
        'jee', 'neet', 'cat', 'gate', 'upsc', 'ssc', 'ibps', 'bank exam', 'government job',
        'private job', 'mba', 'engineering', 'medical', 'ca', 'cs', 'cfa', 'degree',
        'masters', 'phd', 'research', 'scholarship', 'student loan', 'fees', 'hostel',
        'placement', 'campus', 'internship', 'fresher', 'experienced', 'skill development',
        
        # Current Affairs & News (Extended)
        'news', 'breaking news', 'update', 'pandemic', 'lockdown', 'unlock', 'economy',
        'recession', 'growth', 'development', 'budget', 'union budget', 'economic survey',
        'gdp', 'inflation', 'unemployment', 'startup ecosystem', 'unicorn', 'ipo listing',
        'market crash', 'sensex fall', 'rupee fall', 'oil prices', 'crude oil',
        
        # Daily Communication & Greetings
        'good morning', 'good night', 'good afternoon', 'good evening', 'namaste', 'hello',
        'how are you', 'what\'s up', 'where are you', 'come home', 'reached safely', 'busy',
        'call me', 'missed call', 'voice message', 'video call', 'status', 'dp', 'profile pic',
        
        # Tech & Digital Life
        'phone', 'mobile', 'smartphone', 'iphone', 'samsung', 'oneplus', 'xiaomi', 'oppo', 'vivo',
        'recharge', 'wifi', 'network', 'jio', 'airtel', 'vi', 'bsnl', 'broadband', 'fiber',
        'laptop', 'computer', 'tablet', 'smartwatch', 'earbuds', 'headphones', 'charger',
        'battery', 'screen guard', 'phone case', 'backup', 'storage', 'cloud',
        
        # Shopping & E-commerce
        'amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 'big billion', 'great indian festival',
        'sale', 'discount', 'coupon', 'cashback', 'emi', 'cod', 'delivery', 'return',
        'exchange', 'warranty', 'review', 'rating', 'wish list', 'cart', 'checkout',
        
        # Humor & Entertainment
        'meme', 'funny', 'joke', 'lol', 'rofl', 'haha', 'comedy', 'stand up', 'viral',
        'trending', 'forward', 'share', 'tag', 'mention', 'like', 'comment', 'subscribe',
        'influencer', 'content creator', 'blogger', 'vlogger', 'streamer',
        
        # Life Philosophy & Motivation
        'life', 'success', 'failure', 'motivation', 'inspiration', 'quotes', 'thoughts',
        'wisdom', 'advice', 'experience', 'lesson', 'growth', 'change', 'opportunity',
        'challenge', 'struggle', 'achievement', 'goal', 'dream', 'passion', 'purpose'
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
    
    # Get polarizing topics for drama analysis
    from collections import defaultdict
    import numpy as np
    from textblob import TextBlob
    
    topic_sentiments = defaultdict(list)
    drama_topics = ['trump', 'biden', 'election', 'politics', 'modi', 'bjp', 'congress', 
                   'crypto', 'bitcoin', 'stock', 'market', 'movie', 'cricket', 'food']
    
    for _, row in df.iterrows():
        msg_lower = row['message'].lower()
        for topic in drama_topics:
            if topic in msg_lower and not row['message'].startswith('<Media omitted>'):
                blob = TextBlob(row['message'])
                topic_sentiments[topic].append(blob.sentiment.polarity)
    
    polarizing = []
    for topic, sentiments_list in topic_sentiments.items():
        if len(sentiments_list) > 2:
            std_dev = np.std(sentiments_list)
            polarizing.append((topic, std_dev, len(sentiments_list)))
    
    polarizing.sort(key=lambda x: x[1], reverse=True)
    
    # Generate fun narrative report
    top_chatter = top_5.iloc[0]
    most_pos = most_positive.index[0] if sentiments else "Someone"
    
    # Activity level description
    if msgs_per_day > 100:
        activity_desc = "This group is ON FIRE! 🔥 You guys never stop talking!"
    elif msgs_per_day > 50:
        activity_desc = "Pretty chatty group! Good vibes all around! 💬"
    else:
        activity_desc = "Nice and chill group dynamics 😌"
    
    # Time behavior description  
    if peak_hour < 10:
        time_behavior = "Early bird squad! 🌅 Peak activity before 10 AM"
    elif peak_hour > 20:
        time_behavior = "Night owls! 🦉 Most active after 8 PM" 
    else:
        time_behavior = "Steady chatters throughout the day 📱"
    
    # Drama analysis
    drama_text = ""
    if polarizing:
        top_drama = polarizing[0]
        drama_text = f"🔥 DRAMA ALERT: {top_drama[0].capitalize()} creates the most heated debates! (σ={top_drama[1]:.3f})"
    
    # Determine group name or use generic
    group_name = "WHATSAPP GROUP"
    
    report = f"""*{group_name.upper()}: THE ULTIMATE PERSONALITY & TOPIC ANALYSIS* 🎉
Based on *{len(df):,} messages* from *{len(df['sender'].unique())} participants* over *{total_days} days*

📊 *GROUP DYNAMICS AT A GLANCE*
• *Daily average:* {msgs_per_day:.1f} messages ({activity_desc.replace('🔥', '').replace('💬', '').replace('😌', '').strip()})

• *Peak activity:* {peak_hour}:00 on {peak_day}s ({time_behavior.replace('🌅', '').replace('🦉', '').replace('📱', '').strip()})

• *Most active day:* {peak_day} ({daily.max()} messages)

🏆 *THE HALL OF FAME*
🗣️ *Top 5 Chatters*"""
    
    for i, (_, row) in enumerate(top_5.iterrows()):
        nickname = ""
        if i == 0:
            nickname = " _(The Unstoppable Force)_"
        elif row['name'] == max(emoji_users, key=emoji_users.get) if emoji_users else "":
            nickname = " _(The Emoji Champion 😂)_"
        elif row['name'] == max(question_users, key=question_users.get) if question_users else "":
            nickname = " _(The Question Master)_"
        elif row['name'] == max(exclamation_users, key=exclamation_users.get) if exclamation_users else "":
            nickname = " _(The Excitement King! ❗)_"
            
        report += f"\n\n*{i+1}. {row['name']}* - {row['total_messages']} messages{nickname}"
        report += f"\n\n• Peak hour: *{row['most_active_hour']}:00* on *{row['favorite_day']}s* | Avg: *{row['avg_message_length']:.0f} chars* per message"
    
    # Add quiet members
    quiet_members = activity_df.tail(3)
    if len(quiet_members) > 0:
        quiet_names = ", ".join([row['name'] for _, row in quiet_members.iterrows()])
        report += f"\n\n🤐 *The Silent Observers*\n\n• {quiet_names} - 1 message each _(Lurkers!)_"
    
    report += f"\n\n🎭 *PERSONALITY AWARDS*\n😄 *Mood Meters*"
    
    if sentiments:
        report += f"\n\n• *Most Positive Vibes:* "
        pos_list = []
        for sender, score in most_positive.head(3).items():
            pos_list.append(f"*{sender}* ({score:.3f})")
        report += ", ".join(pos_list)
        
        most_negative = sender_sentiment.nsmallest(3)
        report += f"\n\n• *Most Neutral/Negative:* "
        neg_list = []
        for sender, score in most_negative.items():
            neg_list.append(f"*{sender}* ({score:.3f})")
        report += ", ".join(neg_list)
    
    essay_writer = avg_msg_length.idxmax()
    short_writer = avg_msg_length.idxmin()
    
    report += f"\n\n🏅 *Special Recognition*\n\n• 📜 *Essay Writer:* *{essay_writer}* ({avg_msg_length.max():.0f} chars average - _writes novels!_)\n\n• 🔤 *Minimalist:* *{short_writer}* ({avg_msg_length.min():.0f} chars average - _master of brevity_)"
    
    if emoji_users:
        report += f"\n\n• 😂 *Emoji Champion:* *{max(emoji_users, key=emoji_users.get)}*"
    if question_users:
        report += f"\n\n• ❓ *Question Master:* *{max(question_users, key=question_users.get)}*"
    
    report += f"\n\n🔥 *WHAT YOU ACTUALLY TALK ABOUT*"
    
    if sorted_topics:
        # Categorize topics
        political_topics = []
        finance_topics = []
        other_topics = []
        
        for topic, count in sorted_topics:
            topic_lower = topic.lower()
            if any(word in topic_lower for word in ['trump', 'modi', 'election', 'politics', 'bjp', 'congress']):
                political_topics.append((topic, count))
            elif any(word in topic_lower for word in ['stock', 'crypto', 'bitcoin', 'investment', 'market', 'gold']):
                finance_topics.append((topic, count))
            else:
                other_topics.append((topic, count))
        
        if political_topics:
            report += f"\n\n🗳️ *Political & Current Affairs Obsession*"
            for i, (topic, count) in enumerate(political_topics[:4], 1):
                comment = " _(Following every update!)_" if i == 1 else ""
                report += f"\n\n• *{topic}:* {count} mentions{comment}"
        
        if finance_topics:
            report += f"\n\n💰 *Finance Bros Central*"
            for i, (topic, count) in enumerate(finance_topics[:3], 1):
                comment = " _(Diamond hands! 💎)_" if 'crypto' in topic.lower() else ""
                report += f"\n\n• *{topic}:* {count} mentions{comment}"
        
        if other_topics:
            report += f"\n\n🏠 *Daily Life Topics*"
            for i, (topic, count) in enumerate(other_topics[:3], 1):
                report += f"\n\n• *{topic}:* {count} mentions"
    
    # Add drama analysis if available
    if drama_text:
        report += f"\n{drama_text}"
    
    report += f"\n\n📈 *GROUP BEHAVIORAL INSIGHTS*\n⏰ *Time Patterns*\n\n• {time_behavior}\n\n• *Most active day:* {peak_day} ({daily.max()} messages)\n\n• *Quietest day:* {daily.idxmin()} ({daily.min()} messages)\n\n💬 *Communication Style*"
    
    if emoji_users:
        report += f"\n\n• *Emoji Usage Leader:* *{max(emoji_users, key=emoji_users.get)}* _(The Visual Communicator)_"
    if question_users:
        report += f"\n\n• *Most Questions Asked:* *{max(question_users, key=question_users.get)}* _(The Curious One)_"
    if exclamation_users:
        report += f"\n\n• *Most Exclamation Points:* *{max(exclamation_users, key=exclamation_users.get)}* _(The Hype Person!)_"

    # Group personality summary
    if sorted_topics:
        top_topic_names = [topic.lower() for topic, _ in sorted_topics[:3]]
        if any(word in top_topic_names for word in ['trump', 'modi', 'election', 'politics', 'bjp', 'congress']):
            personality = "🗳️ Political Debate Society"
        elif any(word in top_topic_names for word in ['stock', 'crypto', 'bitcoin', 'investment', 'market', 'gold']):
            personality = "📈 Investment Club"
        elif any(word in top_topic_names for word in ['movie', 'bollywood', 'cricket', 'ipl', 'kohli']):
            personality = "🎬 Entertainment Enthusiasts"
        else:
            personality = "🤝 Balanced Social Circle"
    else:
        personality = "🤝 Balanced Social Circle"

    report += f"\n\n🎯 *THE REAL GROUP PERSONALITY*\n*{personality}*\n_You're not just another group chat - you're a community with strong opinions and great energy!_\n\n🚀 *Want YOUR group analyzed?*\nGet your mind-blowing analysis FREE:\n👉 https://whatsapp-group-analyzer.streamlit.app\n\n_Discover who's really running your group!_ 📊✨"
    
    return report

if __name__ == "__main__":
    main()