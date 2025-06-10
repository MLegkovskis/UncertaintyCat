"""
CSS styles for the UncertaintyCat application.
"""

def load_css():
    """
    Returns the CSS styles for the UncertaintyCat application.
    """
    return """
<style>
    .main-header {
        font-family: 'Arial', sans-serif;
        color: #2C3E50;
        padding-bottom: 15px;
        border-bottom: 2px solid #3498DB;
    }
    .sub-header {
        font-family: 'Arial', sans-serif;
        color: #34495E;
        padding: 10px 0;
        margin-top: 20px;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #EBF5FB;
        border-left: 5px solid #3498DB;
    }
    .success-box {
        background-color: #E9F7EF;
        border-left: 5px solid #2ECC71;
    }
    .warning-box {
        background-color: #FEF9E7;
        border-left: 5px solid #F1C40F;
    }
    .error-box {
        background-color: #FDEDEC;
        border-left: 5px solid #E74C3C;
    }
    .stTextArea>div>div>textarea {
        font-family: 'Courier New', monospace;
        background-color: #F8F9FA;
    }
    .section-divider {
        height: 3px;
        background-color: #F0F3F4;
        margin: 30px 0;
        border-radius: 2px;
    }
    /* Custom CSS for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        border: 1px solid #dee2e6;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #3498DB;
        font-weight: bold;
    }
    .tab-content {
        padding: 16px;
        border: 1px solid #dee2e6;
        border-top: none;
        border-radius: 0 0 4px 4px;
    }
</style>
"""
