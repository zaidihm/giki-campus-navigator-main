# GIKI Campus Navigator 🎓

A comprehensive campus navigation system for Ghulam Ishaq Khan Institute (GIKI) that helps students and visitors find their way around the campus. This project combines pathfinding algorithms with natural language processing to provide both navigation assistance and campus information.

## Features ✨

- **Interactive Path Finding**: Find the shortest route between any two locations on campus
- **Multiple Algorithms**: Choose between Dijkstra's and A* algorithms for pathfinding
- **Real-time Map Visualization**: View paths on an interactive map
- **Campus Information**: Ask questions about GIKI and get AI-powered responses
- **User-friendly Interface**: Clean and intuitive design with dark mode support

## Technologies Used 🛠️

- **Backend**: Python
- **Pathfinding**: NetworkX, Dijkstra's Algorithm, A* Algorithm
- **Map Visualization**: Folium
- **Web Interface**: Streamlit
- **AI Integration**: Google Gemini API
- **Data Source**: OpenStreetMap (OSM)

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/giki-campus-navigator.git
cd giki-campus-navigator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Get your free Gemini API key:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Click on "Get API key" to create a new key
   - Copy your API key

5. Create a `.env` file in the project root and add your API key:
```
GOOGLE_API_KEY=your_api_key_here
```

> Note: The Gemini API is currently free to use with generous quotas. You don't need to provide any payment information to get started.

## Usage 📱

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the sidebar to switch between:
   - **Path Finder**: Find routes between locations
   - **Campus Information**: Ask questions about GIKI

## Project Structure 📁

```
giki-campus-navigator/
├── app.py                 # Main Streamlit application
├── graph_algorithms.py    # Graph and pathfinding implementation
├── osm_parser.py         # OSM data parser
├── requirements.txt      # Project dependencies
├── giki.osm             # Campus map data
└── README.md            # Project documentation
```

## Pathfinding Algorithms 📊

### Dijkstra's Algorithm
- Guaranteed to find the shortest path
- Time complexity: O(E + V log V)
- Best for: General pathfinding with weighted edges

### A* Algorithm
- Uses heuristic (Haversine distance) for faster pathfinding
- Still guarantees the shortest path
- Best for: Large graphs with geographical data

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits 👏

Developed by **Nauman Ali Murad (2022479)**,**Hamza Zaidi (2022379)**,**M.Adeel (2022331)** and **Hassan Rais (2022212)** as part of their semester project for Design and Analysis of Algorithms course at Ghulam Ishaq Khan Institute of Engineering Sciences and Technology.

## Acknowledgments 🙏

- OpenStreetMap for providing the campus map data
- Streamlit for the web framework
- NetworkX for graph algorithms 
