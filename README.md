# Python Chess Engine

Overview

This is a high-performance UCI-compatible chess engine written in Python, with an estimated Elo rating of 2400. It features highly optimized search algorithms, deep evaluation capabilities, and strong endgame play using Syzygy tablebases. The engine also includes a massive opening book, precomputed using Stockfish 17 on a powerful 8C/16-thread CPU with 128GB of memory.

Features

UCI-Compatible – Easily integrate with GUI interfaces such as ChessBase, Arena, or CuteChess.

Efficient Evaluation & Fast Searching – Implements alpha-beta pruning, transposition tables, killer moves, history heuristics, and quiescence search.

Deep Search – Quickly reaches high depths for accurate move selection.

Syzygy Tablebases Support – Full support for 3-7 men endgame tablebases.

Comprehensive Opening Book – Built using Stockfish 17, ensuring strong opening play.

Optimized Performance with PyPy – Running with CPython is too slow; PyPy is strongly recommended for significant speed boosts.

Installation

# Clone the repository
git clone https://github.com/yourusername/python-chess-engine.git
cd python-chess-engine

# Install dependencies
pip install -r requirements.txt

Running the Engine

To start the engine in UCI mode:

python engine.py

For optimal performance, run with PyPy:

pypy engine.py

Usage

You can load the engine into a chess GUI like CuteChess or Arena:

Open your chess GUI.

Go to Engine Management.

Add a new UCI engine and select ``.

Start playing!

Future Plans

C++ Version Coming Soon – A highly optimized C++ implementation is in progress.

NNUE Integration – Enhancing evaluation with neural networks.

Parallel Search Implementation – Further improving speed and depth.

Contributions

Contributions, feature requests, and bug reports are welcome! Feel free to open an issue or submit a pull request.

License

This project is licensed under the MIT License.

Contact

For inquiries, reach out via GitHub issues or email me at your-email@example.com.

Note: Performance is significantly better with PyPy. If using CPython, expect much slower search speeds.

