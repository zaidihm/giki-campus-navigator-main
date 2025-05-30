# Campus Navigation System

This project is a Streamlit-based web application that provides an interactive campus navigation system. It uses OpenStreetMap data to create a graph representation of the campus and offers various navigation features.

## Project Overview

The application allows users to:
- View an interactive map of the campus
- Find shortest paths between locations
- Get directions with turn-by-turn navigation
- Search for specific locations
- Visualize the campus network

## Technical Stack

- **Frontend**: Streamlit
- **Mapping**: Folium, OpenStreetMap
- **Graph Processing**: NetworkX
- **Data Processing**: Python
- **AI Integration**: Google Gemini AI
- **Map Data**: OSM (OpenStreetMap)

## Project Structure

```
project/
├── app.py                 # Main Streamlit application
├── graph_algorithms.py    # Graph processing and pathfinding algorithms
├── osm_parser.py         # OpenStreetMap data parser
├── gemini_integration.py # Google Gemini AI integration
├── giki.osm             # Campus map data
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup Instructions

1. **Environment Setup**
   ```bash
   # Create a virtual environment (optional but recommended)
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Dependencies

- networkx==3.2.1
- matplotlib==3.8.2
- streamlit==1.32.0
- google-generativeai==0.3.2
- python-dotenv==1.0.1
- osmium==3.7.0
- folium==0.15.1
- streamlit-folium==0.15.1

## Features

1. **Interactive Map**
   - Zoomable and pannable campus map
   - Location markers and information
   - Custom map styling

2. **Navigation**
   - Shortest path finding
   - Turn-by-turn directions
   - Multiple route options
   - Distance and time estimates

3. **Search Functionality**
   - Location search
   - Building search
   - Category-based filtering

4. **AI Integration**
   - Natural language processing for queries
   - Smart route suggestions
   - Context-aware responses

## Development Steps

1. **Data Collection and Processing**
   - Extracted campus map data from OpenStreetMap
   - Parsed OSM data into a graph structure
   - Implemented data cleaning and validation

2. **Graph Implementation**
   - Created graph representation using NetworkX
   - Implemented pathfinding algorithms
   - Added edge weights and attributes

3. **Web Interface**
   - Developed Streamlit-based UI
   - Integrated interactive map using Folium
   - Implemented user input handling

4. **AI Integration**
   - Set up Google Gemini AI integration
   - Implemented natural language processing
   - Added smart routing features

## Usage

1. Launch the application using `streamlit run app.py`
2. The web interface will open in your default browser
3. Use the search bar to find locations
4. Click on the map to set start and end points
5. View the suggested route and directions
6. Use the AI chat interface for natural language queries

## Future Enhancements

- Real-time traffic updates
- Indoor navigation
- Accessibility features
- Mobile app version
- User preferences and history
- Additional language support

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenStreetMap for map data
- Streamlit for the web framework
- NetworkX for graph processing
- Google Gemini AI for natural language processing