import streamlit as st
from graph_algorithms import CampusGraph
import google.generativeai as genai
from dotenv import load_dotenv
import os
import folium
from streamlit_folium import folium_static
import time
import pandas as pd

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="GIKI Campus Navigator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Google Gemini API
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ Google API Key not found. Please check your .env file.")
    genai.configure(api_key=api_key)
    # Initialize the model - using Gemini Flash for faster responses
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"❌ Error configuring Gemini API: {str(e)}")

# Initialize campus graph
@st.cache_resource
def get_campus_graph():
    return CampusGraph()

campus = get_campus_graph()

def get_gemini_response(question):
    """Get response from Gemini API"""
    try:
        if not question.strip():
            return "Please enter a valid question about GIKI campus."

        prompt = f"""You are a helpful assistant for GIKI (Ghulam Ishaq Khan Institute) campus. 
        Please provide accurate and helpful information about GIKI. Focus on:
        - Academic buildings and departments
        - Student facilities and services
        - Campus landmarks and important locations
        - Student life and activities
        - Campus history and facts
        
        If the question is about navigation or finding specific locations, suggest using the Path Finder tab.
        If you don't know the answer, be honest and say so.
        
        Question: {question}
        
        Answer:"""
        
        response = model.generate_content(prompt)
        if not response.text:
            return "I couldn't generate a response. Please try rephrasing your question."
        return response.text
    except Exception as e:
        return f"Error getting response: {str(e)}. Please check your internet connection and API key."

def get_campus_info(question):
    """Get information about GIKI campus"""
    # Static responses for common questions
    responses = {
        "academic": "GIKI has several academic departments including Computer Science, Electrical Engineering, Mechanical Engineering, and Chemical Engineering. The main academic buildings are located in the central campus area.",
        "facilities": "GIKI provides various facilities including a central library, sports complex, student center, and multiple cafeterias. The campus also has well-equipped laboratories and research centers.",
        "landmarks": "Key landmarks include the Main Gate, Academic Block, Central Library, and the iconic GIKI Tower. The campus is known for its beautiful architecture and green spaces.",
        "student life": "Student life at GIKI is vibrant with numerous clubs, societies, and events. The campus hosts technical festivals, cultural events, and sports competitions throughout the year.",
        "history": "GIKI was established in 1993 and is named after Ghulam Ishaq Khan, the former President of Pakistan. It is known for its excellence in engineering education and research.",
        "default": "I can provide information about GIKI's academic departments, facilities, landmarks, student life, and history. Please ask a specific question about any of these topics."
    }
    
    # Simple keyword matching
    question = question.lower()
    if "academic" in question or "department" in question:
        return responses["academic"]
    elif "facility" in question or "service" in question:
        return responses["facilities"]
    elif "landmark" in question or "location" in question:
        return responses["landmarks"]
    elif "student" in question or "life" in question or "activity" in question:
        return responses["student life"]
    elif "history" in question or "establish" in question:
        return responses["history"]
    else:
        return responses["default"]

def main():
    # Custom CSS
    st.markdown(""" <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        color: #ffffff;
    }

    /* Headers */
    .stMarkdown h1 {
        color: #00ff88;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .stMarkdown h2 {
        color: #00ff88;
        font-size: 1.8rem;
        margin-top: 1.5rem;
        border-bottom: 2px solid #00ff88;
        padding-bottom: 0.5rem;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #1a1a2e;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
    }

    /* Select boxes */
    .stSelectbox>div>div>select {
        background-color: #2a2a3a;
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #00ff88;
        padding: 8px;
    }

    /* Text input */
    .stTextInput>div>div>input {
        background-color: #2a2a3a;
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #00ff88;
        padding: 8px;
    }

    /* Project info box */
    .project-info {
        background: linear-gradient(135deg, #2a2a3a 0%, #1a1a2e 100%);
        color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border: 1px solid #00ff88;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Algorithm info box */
    .algorithm-info {
        background: linear-gradient(135deg, #2a2a3a 0%, #1a1a2e 100%);
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Success message */
    .stSuccess {
        background-color: rgba(0,255,136,0.1);
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 10px;
    }

    /* Error message */
    .stError {
        background-color: rgba(255,0,0,0.1);
        border: 1px solid #ff0000;
        border-radius: 10px;
        padding: 10px;
    }

    /* Warning message */
    .stWarning {
        background-color: rgba(255,165,0,0.1);
        border: 1px solid #ffa500;
        border-radius: 10px;
        padding: 10px;
    }

    /* Loading animation */
    .stSpinner > div {
        border-color: #00ff88 !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    ::-webkit-scrollbar-thumb {
        background: #00ff88;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00cc6a;
    }

    /* Performance bar */
    .performance-bar {
        background: #2a2a3a;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .performance-bar-fill {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
        height: 20px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #00ff88; font-size: 2rem; margin-bottom: 0;">GIKI Campus Navigator</h1>
            <p style="color: #ffffff; font-size: 0.9rem;">Your Smart Campus Guide</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        page = st.radio("Navigation", ["🗺️ Path Finder", "📊 Algorithm Analysis", "❓ Campus Information"])
        st.markdown("---")
        
        st.markdown("### About")
        st.markdown("""
        <div style="color: #ffffff;">
        This app helps you navigate the GIKI campus and get information about various locations.
        
        - Use **Path Finder** to find the shortest route between locations
        - Use **Algorithm Analysis** to compare pathfinding algorithms
        - Use **Campus Information** to ask questions about GIKI
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Project Info")
        st.markdown("""
        <div class="project-info">
        <p style="color: #00ff88; font-weight: bold;">Developed by Nauman A. Murad, Hassan Rais, M. Adeel & Hamza Zaidi</p>
        <p>As part of semester project for Design and Analysis of Algorithms course</p>
        <p>Ghulam Ishaq Khan Institute of Engineering Sciences and Technology</p>
        </div>
        """, unsafe_allow_html=True)

    if page == "🗺️ Path Finder":
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #00ff88; font-size: 2.5rem; margin-bottom: 0;">AI Powered Campus Path Finder for GIKI</h1>
            <p style="color: #ffffff; font-size: 1.1rem;">Find the shortest route between any two locations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get available locations
        locations = list(campus.graph.nodes())
        
        # Create columns for inputs
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            start = st.selectbox("📍 Starting Point", locations, index=0)
        with col2:
            end = st.selectbox("🏁 Destination", locations, index=1)
        with col3:
            algorithm = st.selectbox("🔍 Algorithm", 
                                   ["Dijkstra", "A*", "BFS", "DFS", "Greedy Best-First"], 
                                   help="""Choose pathfinding algorithm:
                                   - Dijkstra: Guaranteed shortest path
                                   - A*: Faster with heuristic
                                   - BFS: Level-by-level search
                                   - DFS: Depth-first exploration
                                   - Greedy: Fast but not optimal""")
        
        if st.button("Find Path 🚀"):
            try:
                with st.spinner("🔍 Finding the best path..."):
                    start_time = time.perf_counter()
                    
                    # Map algorithm names to their internal representations
                    algo_map = {
                        "Dijkstra": "dijkstra",
                        "A*": "astar",
                        "BFS": "bfs",
                        "DFS": "dfs",
                        "Greedy Best-First": "greedy"
                    }
                    
                    # Get path with selected algorithm
                    path, distance = campus.find_path(
                        start, end, 
                        algorithm=algo_map[algorithm],
                        heuristic_type="haversine"  # Default heuristic for A*
                    )
                    execution_time = time.perf_counter() - start_time
                
                # Display results in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ Path found! Total distance: {distance:.2f} meters")
                    st.markdown("### Route:")
                    for i, location in enumerate(path, 1):
                        st.markdown(f"<div style='color: #ffffff;'>{i}. {location}</div>", unsafe_allow_html=True)
                    
                    # Display algorithm info
                    st.markdown("### Algorithm Info:")
                    algo_info = {
                        "Dijkstra": """
                        <div class="algorithm-info">
                        - Dijkstra's Algorithm: Explores all possible paths
                        - Guarantee: Always finds the shortest path
                        - Time Complexity: O(E + V log V)
                        - Space Complexity: O(V)
                        - Best for: Graphs with positive edge weights
                        </div>
                        """,
                        "A*": """
                        <div class="algorithm-info">
                        - A* Algorithm: Uses heuristic to find path faster
                        - Guarantee: Still finds the shortest path
                        - Time Complexity: O(E) in best case, O(E + V log V) in worst case
                        - Space Complexity: O(V)
                        - Best for: Large graphs with good heuristics
                        </div>
                        """,
                        "BFS": """
                        <div class="algorithm-info">
                        - Breadth-First Search: Explores level by level
                        - Guarantee: Finds shortest path in unweighted graphs
                        - Time Complexity: O(V + E)
                        - Space Complexity: O(V)
                        - Best for: Unweighted graphs, finding minimum steps
                        </div>
                        """,
                        "DFS": """
                        <div class="algorithm-info">
                        - Depth-First Search: Explores as far as possible
                        - Guarantee: Not guaranteed to find shortest path
                        - Time Complexity: O(V + E)
                        - Space Complexity: O(V)
                        - Best for: Exploring all possible paths
                        </div>
                        """,
                        "Greedy Best-First": """
                        <div class="algorithm-info">
                        - Greedy Best-First: Always chooses best heuristic
                        - Guarantee: Not guaranteed to find shortest path
                        - Time Complexity: O(V + E)
                        - Space Complexity: O(V)
                        - Best for: Quick approximate solutions
                        </div>
                        """
                    }
                    st.markdown(algo_info[algorithm], unsafe_allow_html=True)
                    
                    st.markdown(f"### Performance Metrics")
                    st.markdown(f"""
                    <div class="algorithm-info">
                    - Execution Time: {execution_time:.6f} seconds
                    - Path Length: {len(path)} nodes
                    - Total Distance: {distance:.2f} meters
                    - Nodes Explored: {len(campus.graph.nodes())}
                    - Obstacles Avoided: {len(campus.obstacles)}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Map View")
                    m = campus.visualize_path(path)
                    folium_static(m)
                
            except Exception as e:
                st.error(f"❌ Error finding path: {str(e)}")

    elif page == "📊 Algorithm Analysis":
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #00ff88; font-size: 2.5rem; margin-bottom: 0;">📊 Algorithm Analysis</h1>
            <p style="color: #ffffff; font-size: 1.1rem;">Compare the performance of pathfinding algorithms</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get available locations
        locations = list(campus.graph.nodes())
        
        col1, col2 = st.columns(2)
        with col1:
            start = st.selectbox("📍 Starting Point", locations, index=0, key="analysis_start")
        with col2:
            end = st.selectbox("🏁 Destination", locations, index=1, key="analysis_end")
        
        if st.button("Compare Algorithms 🚀", key="compare_algo"):
            try:
                with st.spinner("🔍 Analyzing algorithms..."):
                    algorithms = {
                        "Dijkstra": {"algo": "dijkstra", "heuristic": None},
                        "A*": {"algo": "astar", "heuristic": "haversine"},
                        "BFS": {"algo": "bfs", "heuristic": None},
                        "DFS": {"algo": "dfs", "heuristic": None},
                        "Greedy Best-First": {"algo": "greedy", "heuristic": None}
                    }
                    
                    results = []
                    for name, config in algorithms.items():
                        start_time = time.perf_counter()
                        if config["heuristic"]:
                            path, distance = campus.find_path(start, end, algorithm=config["algo"], heuristic_type=config["heuristic"])
                        else:
                            path, distance = campus.find_path(start, end, algorithm=config["algo"])
                        execution_time = time.perf_counter() - start_time
                        
                        results.append({
                            "Algorithm": name,
                            "Execution Time (s)": execution_time,
                            "Path Length": len(path),
                            "Total Distance (m)": distance,
                            "Path": path
                        })
                
                # Display comparison results
                st.markdown("### 📈 Algorithm Comparison Metrics")
                
                # Create dataframe with formatted values for display
                display_df = pd.DataFrame(results)
                display_df["Execution Time (s)"] = display_df["Execution Time (s)"].apply(lambda x: f"{x:.6f}")
                display_df["Total Distance (m)"] = display_df["Total Distance (m)"].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(
                    display_df[["Algorithm", "Execution Time (s)", "Path Length", "Total Distance (m)"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Algorithm": st.column_config.TextColumn("Algorithm", width="medium"),
                        "Execution Time (s)": st.column_config.TextColumn("Time (s)"),
                        "Path Length": st.column_config.NumberColumn("Nodes"),
                        "Total Distance (m)": st.column_config.TextColumn("Distance (m)")
                    }
                )
                
                # Add visual metrics
                st.markdown("### ⚡ Performance Highlights")
                col1, col2, col3 = st.columns(3)
                
                fastest = min(results, key=lambda x: x["Execution Time (s)"])
                shortest = min(results, key=lambda x: x["Total Distance (m)"])
                most_nodes = max(results, key=lambda x: x["Path Length"])
                
                with col1:
                    st.metric("Fastest Algorithm", fastest["Algorithm"], f"{fastest['Execution Time (s)']:.6f}s")
                with col2:
                    st.metric("Shortest Distance", shortest["Algorithm"], f"{shortest['Total Distance (m)']:.2f}m")
                with col3:
                    st.metric("Most Nodes Explored", most_nodes["Algorithm"], f"{most_nodes['Path Length']} nodes")
                
                # Display paths in a grid layout
                st.markdown("### 🗺️ Path Visualizations")
                
                for i in range(0, len(results), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(results):
                            with cols[j]:
                                result = results[i + j]
                                st.markdown(f"#### {result['Algorithm']}")
                                
                                with st.container():
                                    m = campus.visualize_path(result['Path'])
                                    folium_static(m, width=300, height=300)
                                    
                                st.caption(f"Distance: {result['Total Distance (m)']:.2f}m | Time: {result['Execution Time (s)']:.6f}s")
                
                # Add analysis
                st.markdown("### 📝 Analysis Summary")
                st.markdown("""
                <div style="background-color: #2a2a3a; padding: 15px; border-radius: 10px; border-left: 4px solid #00ff88;">
                <p><b>Key Observations:</b></p>
                <ul>
                    <li>Dijkstra and A* typically find the shortest paths</li>
                    <li>A* is usually faster than Dijkstra due to heuristic</li>
                    <li>BFS finds shortest paths in unweighted graphs</li>
                    <li>DFS may find longer paths as it explores depth-first</li>
                    <li>Greedy algorithms are fast but may not be optimal</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error comparing algorithms: {str(e)}")

    elif page == "❓ Campus Information":
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #00ff88; font-size: 2.5rem; margin-bottom: 0;">❓ Ask About GIKI</h1>
            <p style="color: #ffffff; font-size: 1.1rem;">Get AI-powered answers about the campus</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Text input for questions
        question = st.text_input("Ask any question about GIKI:", 
                               placeholder="Example: What are the main academic buildings?")
        
        if st.button("Get Answer 💡"):
            if question:
                with st.spinner("🤔 Getting answer..."):
                    response = get_gemini_response(question)
                    st.markdown("### Answer:")
                    st.markdown(f"<div style='color: #ffffff;'>{response}</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter a question.")

if __name__ == "__main__":
    main()