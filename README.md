AI Traffic Signal Optimization System
This project is an AI-based Traffic Signal Optimization System designed to improve traffic flow efficiency using reinforcement learning and other AI techniques.

Features
Dynamic Traffic Signal Control: Optimizes traffic signals in real-time based on traffic conditions.
Reinforcement Learning Agents: Includes Q-learning and fixed-time agents for traffic management.
Evaluation Tools: Tools to evaluate the performance of the system.
Visualization: Graphical representation of training history and evaluation results.
Frontend Integration: A simple frontend for visualization and interaction.
Project Structure

traffic_ai/
├── agent/               # Contains agent implementations (e.g., Q-learning, fixed-time)
├── environment/         # Traffic environment simulation
├── evaluation/          # Evaluation scripts and tools
├── frontend/            # Frontend files for visualization
├── outputs/             # Stores evaluation summaries and training history
├── plots/               # Visualization scripts and generated plots
├── training/            # Training scripts for the agents
├── fix_offline.py       # Script for offline fixes
├── main.py              # Entry point for the system
├── server.py            # Backend server for the system
└── README.md            # Project documentation

Installation
Clone the repository:
git clone https://github.com/yagamieren09/AI-Traffic-Signal-Optimization-System.git
cd AI-Traffic-Signal-Optimization-System

Install the required dependencies:
pip install -r requirements.txt

Run the server:
python server.py

Access the frontend in your browser at http://localhost:5000.

Usage
Training: Use the train.py script in the training/ directory to train the agents.
Evaluation: Use the evaluate.py script in the evaluation/ directory to evaluate the system's performance.
Visualization: Use the visualize.py script in the plots/ directory to generate plots.


