import json
import os
from datetime import datetime
import hashlib
import streamlit as st

def get_client_info():
    """Get anonymized client information for analytics"""
    try:
        # Get basic info from Streamlit context (available info is limited)
        session_info = st.runtime.get_instance().get_client()
        
        # Create anonymous user hash (changes each session, no persistent tracking)
        user_agent = st.context.headers.get('user-agent', 'unknown') if hasattr(st, 'context') and hasattr(st.context, 'headers') else 'unknown'
        session_id = hashlib.sha256(f"{datetime.now().date()}{user_agent}".encode()).hexdigest()[:12]
        
        return {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'user_agent': user_agent[:100] if user_agent != 'unknown' else 'unknown',  # Truncate for privacy
        }
    except Exception as e:
        # Fallback for minimal tracking
        return {
            'session_id': hashlib.sha256(f"{datetime.now()}".encode()).hexdigest()[:12],
            'timestamp': datetime.now().isoformat(),
            'user_agent': 'unknown'
        }

def log_usage_event(event_type, **kwargs):
    """Log a usage event to analytics file"""
    try:
        analytics_file = 'usage_analytics.json'
        
        # Prepare event data
        event_data = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            **get_client_info(),
            **kwargs  # Additional event-specific data
        }
        
        # Load existing data
        if os.path.exists(analytics_file):
            try:
                with open(analytics_file, 'r') as f:
                    analytics_data = json.load(f)
            except:
                analytics_data = []
        else:
            analytics_data = []
        
        # Add new event
        analytics_data.append(event_data)
        
        # Keep only last 1000 events to prevent file from growing too large
        if len(analytics_data) > 1000:
            analytics_data = analytics_data[-1000:]
        
        # Save updated data
        with open(analytics_file, 'w') as f:
            json.dump(analytics_data, f, indent=2)
            
    except Exception as e:
        # Fail silently to not break the main app
        print(f"Analytics logging failed: {e}")

def get_usage_stats():
    """Get usage statistics from analytics file"""
    try:
        analytics_file = 'usage_analytics.json'
        
        if not os.path.exists(analytics_file):
            return {
                'total_sessions': 0,
                'total_analyses': 0,
                'recent_activity': [],
                'daily_stats': {}
            }
        
        with open(analytics_file, 'r') as f:
            analytics_data = json.load(f)
        
        # Calculate stats
        page_views = [event for event in analytics_data if event['event_type'] == 'page_view']
        analyses = [event for event in analytics_data if event['event_type'] == 'analysis_completed']
        
        # Daily breakdown
        daily_stats = {}
        for event in analytics_data:
            date = event['timestamp'][:10]  # Extract YYYY-MM-DD
            if date not in daily_stats:
                daily_stats[date] = {'views': 0, 'analyses': 0}
            
            if event['event_type'] == 'page_view':
                daily_stats[date]['views'] += 1
            elif event['event_type'] == 'analysis_completed':
                daily_stats[date]['analyses'] += 1
        
        return {
            'total_sessions': len(page_views),
            'total_analyses': len(analyses),
            'recent_activity': analytics_data[-20:],  # Last 20 events
            'daily_stats': daily_stats
        }
        
    except Exception as e:
        print(f"Failed to get usage stats: {e}")
        return {
            'total_sessions': 0,
            'total_analyses': 0,
            'recent_activity': [],
            'daily_stats': {}
        }

def display_analytics_dashboard():
    """Display analytics dashboard (for admin use)"""
    st.subheader("📊 Usage Analytics Dashboard")
    
    stats = get_usage_stats()
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sessions", stats['total_sessions'])
    with col2:
        st.metric("Total Analyses", stats['total_analyses'])
    with col3:
        conversion_rate = (stats['total_analyses'] / max(stats['total_sessions'], 1)) * 100
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    
    # Daily activity chart
    if stats['daily_stats']:
        st.subheader("Daily Activity")
        import pandas as pd
        
        daily_df = pd.DataFrame.from_dict(stats['daily_stats'], orient='index')
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df.sort_index()
        
        st.line_chart(daily_df)
    
    # Recent activity
    st.subheader("Recent Activity")
    if stats['recent_activity']:
        for event in reversed(stats['recent_activity'][-10:]):  # Show last 10
            event_time = datetime.fromisoformat(event['timestamp']).strftime('%Y-%m-%d %H:%M')
            st.text(f"{event_time} - {event['event_type']} - Session: {event['session_id']}")
    else:
        st.text("No recent activity")