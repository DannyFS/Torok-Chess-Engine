import chess
import time
import hashlib

# Piece values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 305,
    chess.BISHOP: 333,
    chess.ROOK: 563,
    chess.QUEEN: 950,
    chess.KING: 20000
}

# Piece-Square Tables
PAWN_PST = [
    0,   5,  5,   0,   5,  10,  50,  0,
    0,  10, -5,   0,   5,  10,  50,  0,
    0,  10,  10, 20,  25,  30,  50,  0,
    0,  10,  20, 35,  40,  45,  50,  0,
    0,  10,  20, 35,  40,  45,  50,  0,
    0,  10,  10, 20,  25,  30,  50,  0,
    0,  10, -5,   0,   5,  10,  50,  0,
    0,   5,  5,   0,   5,  10,  50,  0
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

ROOK_PST = [
    0,   0,   5,  10,  10,   5,   0,   0,
    0,   0,   5,  10,  10,   5,   0,   0,
    0,   0,   5,  10,  10,   5,   0,   0,
    0,   0,   5,  10,  10,   5,   0,   0,
    0,   0,   5,  10,  10,   5,   0,   0,
    0,   0,   5,  10,  10,   5,   0,   0,
    0,   0,   5,  10,  10,   5,   0,   0,
    0,   0,   5,  10,  10,   5,   0,   0
]

QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,  -5, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -5,    0,   5,   5,   5,   5,   0,  -5,
    -5,    0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,  -5, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20
]

KING_PST = [
    20,  30,  10,   0,   0,  10,  30,  20,
    20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST
}

class ChessEngine:
    def __init__(self):
        self.nodes_searched = 0
        self.best_move = None
        self.transposition_table = {}  
        self.killer_moves = {}  
        self.history_heuristic = {}  

    def evaluate_board(self, board):
        """Evaluate board using material balance, piece-square tables, king safety, pawn structure, and mobility."""
        score = 0

        # Material Balance
        for piece_type in PIECE_VALUES:
            for square in board.pieces(piece_type, chess.WHITE):
                score += PIECE_VALUES[piece_type] + PIECE_SQUARE_TABLES[piece_type][square]
            for square in board.pieces(piece_type, chess.BLACK):
                score -= PIECE_VALUES[piece_type] + PIECE_SQUARE_TABLES[piece_type][chess.square_mirror(square)]

        # King Safety
        score += self.king_safety(board, chess.WHITE)
        score -= self.king_safety(board, chess.BLACK)

        # Pawn Structure
        score += self.evaluate_pawn_structure(board, chess.WHITE)
        score -= self.evaluate_pawn_structure(board, chess.BLACK)

        # Mobility
        score += self.evaluate_mobility(board, chess.WHITE)
        score -= self.evaluate_mobility(board, chess.BLACK)
        
        #Passed Pawns
        score += self.evaluate_passed_pawns(board, chess.WHITE)
        score -= self.evaluate_passed_pawns(board, chess.BLACK)
        
        # Center Control
        score += self.evaluate_center_control(board, chess.WHITE)
        score -= self.evaluate_center_control(board, chess.BLACK)
        
        #  Connectivity
        score += self.evaluate_connectivity(board, chess.WHITE)
        score -= self.evaluate_connectivity(board, chess.BLACK)
        
        # Trapped pieces
        score -= self.evaluate_trapped_pieces(board, chess.WHITE)
        score += self.evaluate_trapped_pieces(board, chess.BLACK)
        
        # Space
        score += self.evaluate_space(board, chess.WHITE)
        score -= self.evaluate_space(board, chess.BLACK)
        
        # Tempo
        score += self.evaluate_tempo(board, chess.WHITE)
        score -= self.evaluate_tempo(board, chess.BLACK)
        
        # Patterns
        score += self.evaluate_patterns(board, chess.WHITE)
        score -= self.evaluate_patterns(board, chess.BLACK)
        
        # Piece Vulnerability
        score -= self.evaluate_piece_vulnerability(board, chess.WHITE)
        score += self.evaluate_piece_vulnerability(board, chess.BLACK)

        return score


    def king_safety(self, board, color):
        """Evaluate the safety of the king based on castling, pawn shield, and open files."""
        score = 0
        king_square = board.king(color)

        # Encourage castling
        if board.has_castling_rights(color):
            score += 30  # Bonus for having the ability to castle

        # Penalize king in the center (before castling)
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
    
        if king_file in [3, 4] and king_rank in [0, 7]:
            score -= 20  # King is still in the center

        # Pawn shield bonus
        pawn_bonus = 10
        rank_offset = 1 if color == chess.WHITE else -1  # Pawns are ahead of the king

        if 0 <= king_rank + rank_offset <= 7:  # Ensure rank is within bounds
            for file_offset in [-1, 0, 1]:  # Check three squares in front of the king
                file = king_file + file_offset
                if 0 <= file <= 7:  # Ensure file is within bounds
                    pawn_square = chess.square(file, king_rank + rank_offset)
                    piece = board.piece_at(pawn_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        score += pawn_bonus  # Pawn shield bonus

        # Open file penalty (check if there's a pawn in front)
        open_file_penalty = 15
        for file_offset in [-1, 0, 1]:  # Check left, center, and right files near the king
            file = king_file + file_offset
            if 0 <= file <= 7 and 0 <= king_rank + rank_offset <= 7:  # Ensure valid square
                pawn_square = chess.square(file, king_rank + rank_offset)
                if board.piece_at(pawn_square) is None:  # No pawn blocking the file
                    score -= open_file_penalty  

        # King centralization in endgame
        if len(board.piece_map()) < 10:  # If few pieces remain
            center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
            if king_square in center_squares:
                score += 20  # Bonus for centralized king in endgame

        return score

    def evaluate_pawn_structure(self, board, color):
        """Evaluate pawn structure including isolated, doubled, and passed pawns."""
        score = 0
        pawn_squares = list(board.pieces(chess.PAWN, color))
        files = [chess.square_file(sq) for sq in pawn_squares]
    
        # Count pawns in each file
        file_count = {f: 0 for f in range(8)}
        for f in files:
            file_count[f] += 1

        for square in pawn_squares:
            file = chess.square_file(square)
            rank = chess.square_rank(square)

            # **Doubled Pawn Penalty**
            if file_count[file] > 1:
                score -= 15  # Penalize doubled pawns

            # **Isolated Pawn Penalty**
            if (file - 1 not in file_count or file_count[file - 1] == 0) and (file + 1 not in file_count or file_count[file + 1] == 0):
                score -= 20  # No supporting pawns on adjacent files

            # **Passed Pawn Bonus**
            if self.is_passed_pawn(board, square, color):
                score += 30 + (rank * 5 if color == chess.WHITE else (7 - rank) * 5)  # Reward more if closer to promotion

        return score

    def is_passed_pawn(self, board, square, color):
        """Check if a pawn is passed (has no opposing pawns blocking its path to promotion)."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        for f in [file - 1, file, file + 1]:  # Check the pawn's file and adjacent files
            if 0 <= f <= 7:
                for r in range(rank + 1, 8) if color == chess.WHITE else range(rank - 1, -1, -1):
                    if board.piece_at(chess.square(f, r)) and board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and board.piece_at(chess.square(f, r)).color != color:
                        return False  # Enemy pawn blocking its advance

        return True

    def evaluate_mobility(self, board, color):
        """Evaluate mobility by counting legal moves for each piece type."""
        score = 0
        mobility_bonus = {
            chess.PAWN: 0.1,
            chess.KNIGHT: 0.5,
            chess.BISHOP: 0.5,
            chess.ROOK: 0.7,
            chess.QUEEN: 1.0,
            chess.KING: 0.3  # King should move less in the opening/middle game
        }

        for piece_type in mobility_bonus:
            for square in board.pieces(piece_type, color):
                legal_moves = [move for move in board.legal_moves if move.from_square == square]
                score += len(legal_moves) * mobility_bonus[piece_type]

        return score

    def evaluate_passed_pawns(self, board, color):
        """Evaluate passed pawns by rewarding them based on their progress towards promotion."""
        score = 0
        pawn_squares = list(board.pieces(chess.PAWN, color))
    
        for square in pawn_squares:
            if self.is_passed_pawn(board, square, color):
                rank = chess.square_rank(square)
                distance_to_promotion = 7 - rank if color == chess.WHITE else rank
                score += 20 + (6 - distance_to_promotion) * 10  # Reward closer pawns more

        return score

    def evaluate_center_control(self, board, color):
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        bonus_per_attacker = 20  # bonus for each friendly attacker
        penalty_per_attacker = 20  # penalty for each enemy attacker
        score = 0

        for square in center_squares:
            # Count friendly attackers
            my_attackers = board.attackers(color, square)
            # Count enemy attackers
            opponent_attackers = board.attackers(not color, square)
            score += len(my_attackers) * bonus_per_attacker
            score -= len(opponent_attackers) * penalty_per_attacker

        return score

    def evaluate_connectivity(self, board, color):
        """
        Evaluate connectivity among friendly pieces.
        For each piece, add a bonus for every friendly piece defending its square.
        """
        connectivity_bonus = 5  # Adjust this bonus value as needed
        connectivity_score = 0

        # Loop over all piece types except the king (or include with a lower weight)
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                # Count the number of friendly pieces defending this square
                defenders = board.attackers(color, square)
                connectivity_score += len(defenders) * connectivity_bonus

        # Optionally, add a smaller bonus for the king's connectivity
        king_square = board.king(color)
        if king_square is not None:
            connectivity_score += len(board.attackers(color, king_square)) * (connectivity_bonus // 2)

        return connectivity_score
    
    def evaluate_trapped_pieces(self, board, color):
        """
        Evaluate trapped pieces for the given color.
        A piece is considered trapped if it has very few legal moves (or none)
        and is under enemy attack without sufficient friendly support.
        Only non-pawn, non-king pieces are considered.
        """
        penalty = 50  # Base penalty for a trapped piece (adjust as needed)
        total_penalty = 0

        # Consider pieces that are often vulnerable when trapped
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                # Get all legal moves for this piece from its current square.
                moves = [move for move in board.legal_moves if move.from_square == square]
                num_moves = len(moves)
                # Check if the piece is under enemy attack
                under_attack = board.is_attacked_by(not color, square)
                # Check if the piece is defended by a friendly piece
                defended = board.is_attacked_by(color, square)

                # If the piece has no legal moves, it's definitely trapped.
                if num_moves == 0:
                    total_penalty += penalty
                # If the piece is under attack, not defended, and has very limited mobility,
                # consider it partially trapped.
                elif under_attack and not defended and num_moves < 2:
                    total_penalty += penalty // 2

        return total_penalty
    
    def evaluate_space(self, board, color):
        """
        Evaluate space by rewarding control of squares in the opponent's territory.
    
        For White, enemy territory consists of squares on ranks 5-8 (indices 4-7).
        For Black, enemy territory consists of squares on ranks 1-4 (indices 0-3).
    
        Each square controlled in enemy territory contributes a bonus.
        """
        bonus_per_square = 5  # Adjust this value to change the weight of space control
        space_score = 0

        # Define enemy territory based on color
        if color == chess.WHITE:
            enemy_ranks = range(4, 8)  # ranks 5-8
        else:
            enemy_ranks = range(0, 4)  # ranks 1-4

        # Iterate over all board squares
        for square in chess.SQUARES:
            if chess.square_rank(square) in enemy_ranks:
                # If the square is attacked by the given color, add the bonus
                if board.is_attacked_by(color, square):
                    space_score += bonus_per_square

        return space_score
    
    def evaluate_tempo(self, board, color):
        tempo_bonus = 10  # Tune this value for your engine
        return tempo_bonus if board.turn == color else 0
    
    def evaluate_patterns(self, board, color):
        """
        Evaluate positional patterns such as:
        - Bishop pair bonus.
        - Rook on the opponent's back rank (7th rank for White, 2nd for Black).
        - Knight outpost bonus in the center.
    
        Adjust bonus values as needed.
        """
        pattern_score = 0

        # Bishop Pair Bonus: Having two or more bishops is a well-known advantage.
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            pattern_score += 50  # Bonus for bishop pair

        # Rook on the 7th rank (White) or 2nd rank (Black)
        # For White, ranks are 5th-8th (indices 4-7) but we focus on the 7th rank here.
        if color == chess.WHITE:
            seventh_rank = range(chess.A7, chess.H7 + 1)  # squares on rank 7
        else:
            seventh_rank = range(chess.A2, chess.H2 + 1)  # squares on rank 2
        for square in board.pieces(chess.ROOK, color):
            if square in seventh_rank:
                pattern_score += 25  # Bonus for an active rook on opponent's back rank

        # Knight Outpost Bonus: Reward knights placed on central squares where they are less likely to be challenged.
        knight_bonus = 15
        if color == chess.WHITE:
            center_outposts = {chess.C3, chess.D3, chess.E3, chess.F3, chess.C4, chess.D4, chess.E4, chess.F4}
        else:
            center_outposts = {chess.C5, chess.D5, chess.E5, chess.F5, chess.C6, chess.D6, chess.E6, chess.F6}
        for square in board.pieces(chess.KNIGHT, color):
            if square in center_outposts:
                pattern_score += knight_bonus

        return pattern_score
    
    def evaluate_piece_vulnerability(self, board, color):
        """
        Evaluate the vulnerability of pieces by penalizing those that are
        attacked by the opponent more than they are defended.
        """
        penalty = 0
        # Evaluate all pieces except the king (which is handled by king safety)
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                attackers = board.attackers(not color, square)
                defenders = board.attackers(color, square)
                # If attackers outnumber defenders, assign a penalty based on the piece's value.
                if len(attackers) > len(defenders):
                    imbalance = len(attackers) - len(defenders)
                    penalty += PIECE_VALUES[piece_type] * imbalance * 0.1  # adjust multiplier as needed
        return penalty


    def hash_board(self, board):
        """Generate a unique hash for a given board state."""
        board_fen = board.fen()  
        return hashlib.sha256(board_fen.encode('utf-8')).hexdigest()

    def order_moves(self, board, legal_moves, depth):
        """Order moves to improve alpha-beta pruning efficiency."""
        scored_moves = []
        for move in legal_moves:
            score = 0
            if move == self.best_move:
                score += 100000
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim:
                    score += 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[board.piece_at(move.from_square).piece_type]
            if self.killer_moves.get(depth) == move:
                score += 5000
            score += self.history_heuristic.get(move, 0)
            scored_moves.append((score, move))

        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored_moves]

    def quiescence_search(self, board, alpha, beta, color):
        """Quiescence search for evaluating tactical positions beyond regular depth."""
        self.nodes_searched += 1
        stand_pat = color * self.evaluate_board(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                board.push(move)
                eval = -self.quiescence_search(board, -beta, -alpha, -color)
                board.pop()
                if eval >= beta:
                    return beta
                if eval > alpha:
                    alpha = eval
        return alpha

    def negamax(self, board, depth, alpha, beta, color):
        """NegaMax with alpha-beta pruning, transposition table, and quiescence search."""
        self.nodes_searched += 1
        board_hash = self.hash_board(board)
        if board_hash in self.transposition_table:
            transposition = self.transposition_table[board_hash]
            if transposition['depth'] >= depth:
                return transposition['score']
        if depth == 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta, color)
        best_eval = float('-inf')
        ordered_moves = self.order_moves(board, list(board.legal_moves), depth)
        for move in ordered_moves:
            board.push(move)
            eval = -self.negamax(board, depth - 1, -beta, -alpha, -color)
            board.pop()
            if eval > best_eval:
                best_eval = eval
                if depth == self.max_depth:
                    self.best_move = move
            alpha = max(alpha, eval)
            if alpha >= beta:
                self.killer_moves[depth] = move
                break
        self.transposition_table[board_hash] = {'score': best_eval, 'depth': depth}
        return best_eval

    def find_best_move(self, board, max_depth=5):
        """Find best move using Iterative Deepening with Move Ordering and Quiescence Search."""
        self.nodes_searched = 0
        self.best_move = None
        self.max_depth = max_depth
        start_time = time.time()
        for depth in range(1, max_depth + 1):
            self.negamax(board, depth, float('-inf'), float('inf'), 1)
        time_taken = time.time() - start_time
        nps = self.nodes_searched / time_taken if time_taken > 0 else 0
        print("\n--- Engine Analysis ---")
        print(f"Best Move: {self.best_move}")
        print(f"Nodes Searched: {self.nodes_searched}")
        print(f"Max Depth Reached: {max_depth}")
        print(f"Time Taken: {time_taken:.3f} sec")
        print(f"Nodes Per Second: {int(nps)}\n")
        return self.best_move

# Play against the engine
board = chess.Board()
engine = ChessEngine()

while not board.is_game_over():
    print(board)
    if board.turn == chess.WHITE:
        print("Engine thinking...\n")
        best_move = engine.find_best_move(board, max_depth=4)
        if best_move:
            board.push(best_move)
        else:
            print("No valid moves! Engine resigns.")
            break
    else:
        move = input("Enter your move (UCI format, e.g., e2e4): ")
        if chess.Move.from_uci(move) in board.legal_moves:
            board.push(chess.Move.from_uci(move))
        else:
            print("Invalid move, try again.")
            continue

print("\nGame Over!")
print("Result:", board.result())