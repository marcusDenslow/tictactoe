import time
import random
import numpy as np
from math import inf as infinity

class TicTacToe:
    def __init__(self, size=3):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # Player 1 (X) starts
        self.player_symbols = {0: ' ', 1: 'X', -1: 'O'}
        self.game_over = False
        self.winner = None
        
        # For larger boards (>3), we need different win conditions
        if size <= 3:
            self.win_length = size  # Classic 3-in-a-row for 3x3
        elif size <= 5:
            self.win_length = 4  # Need 4-in-a-row for 4x4 and 5x5
        else:
            self.win_length = 5  # Need 5-in-a-row for 6x6 and larger
    
    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
    
    def change_player(self):
        self.current_player *= -1
    
    def make_move(self, row, col):
        if self.board[row, col] == 0 and not self.game_over:
            self.board[row, col] = self.current_player
            self.check_game_over()
            if not self.game_over:
                self.change_player()
            return True
        return False
    
    def check_game_over(self):
        # Check if anyone has won
        win_length = self.win_length
        size = self.size
        board = self.board
        
        # Check rows
        for r in range(size):
            for c in range(size - win_length + 1):
                if board[r, c] != 0:
                    if np.all(board[r, c:c+win_length] == board[r, c]):
                        self.game_over = True
                        self.winner = board[r, c]
                        return
        
        # Check columns
        for c in range(size):
            for r in range(size - win_length + 1):
                if board[r, c] != 0:
                    if np.all(board[r:r+win_length, c] == board[r, c]):
                        self.game_over = True
                        self.winner = board[r, c]
                        return
        
        # Check diagonals (top-left to bottom-right)
        for r in range(size - win_length + 1):
            for c in range(size - win_length + 1):
                if board[r, c] != 0:
                    diagonal = True
                    for i in range(1, win_length):
                        if board[r+i, c+i] != board[r, c]:
                            diagonal = False
                            break
                    if diagonal:
                        self.game_over = True
                        self.winner = board[r, c]
                        return
        
        # Check diagonals (top-right to bottom-left)
        for r in range(size - win_length + 1):
            for c in range(win_length - 1, size):
                if board[r, c] != 0:
                    diagonal = True
                    for i in range(1, win_length):
                        if board[r+i, c-i] != board[r, c]:
                            diagonal = False
                            break
                    if diagonal:
                        self.game_over = True
                        self.winner = board[r, c]
                        return
        
        # Check for draw
        if np.all(board != 0):
            self.game_over = True
            self.winner = 0
    
    def display_board(self):
        size = self.size
        print()
        # Print column numbers
        print('   ', end='')
        for col in range(size):
            print(f' {col + 1}  ', end='')
        print()
        
        # Print top border
        print('  ' + '+' + '---+' * size)
        
        # Print board rows
        for row in range(size):
            print(f'{row + 1} |', end='')
            for col in range(size):
                symbol = self.player_symbols[self.board[row, col]]
                print(f' {symbol} |', end='')
            print()
            print('  ' + '+' + '---+' * size)
        print()
    
    def get_available_moves(self):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0:
                    moves.append((r, c))
        return moves
    
    def copy(self):
        """Create a deep copy of this game state"""
        new_game = TicTacToe(self.size)
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game


class ScalableTicTacToeAI:
    """An AI optimized for boards of any size with advanced tactical awareness"""
    
    def __init__(self, game, max_depth=None, max_time=None):
        self.game = game
        self.max_depth = max_depth if max_depth is not None else infinity
        self.max_time = max_time
        self.start_time = None
        self.time_up = False
        
        # Cache for position evaluation
        self.evaluation_cache = {}
        
        # Adjust search parameters based on board size for optimal performance
        self.size = game.size
        self.win_length = game.win_length
        
        # Adaptive search depth based on board size
        if self.size <= 3:
            self.adaptive_depth = 9  # Full search for 3x3
        elif self.size <= 4:
            self.adaptive_depth = 6  # Reduced for 4x4
        elif self.size <= 5:
            self.adaptive_depth = 4  # Further reduced for 5x5
        else:
            self.adaptive_depth = 3  # Minimal for larger boards
        
        # Counter-threat parameters
        self.enable_counter_threats = True        
        
        # Debug mode for detailed output
        self.debug = False
    
    def get_move(self):
        """Find the best move for the current player"""
        available_moves = self.game.get_available_moves()
        
        if not available_moves:
            return None
        
        player = self.game.current_player
        opponent = -player
        board = self.game.board
        
        if self.debug:
            print(f"AI starting move selection (board size: {self.size}, win length: {self.win_length})")
        
        # Special case for first moves on large boards
        if np.count_nonzero(board) <= 1 and self.size >= 4:
            center = self.size // 2
            if board[center, center] == 0:
                if self.debug:
                    print("Taking center position")
                return (center, center)
            # Take near center if center is taken
            for r in range(center-1, center+2):
                for c in range(center-1, center+2):
                    if 0 <= r < self.size and 0 <= c < self.size and board[r, c] == 0:
                        return (r, c)
        
        # ------------------- TACTICAL DECISION HIERARCHY -------------------
        
        # 1. IMMEDIATE WIN: Can we win now?
        winning_move = self.find_winning_move(player)
        if winning_move:
            if self.debug:
                print(f"Found winning move: {winning_move}")
            return winning_move
        
        # 2. IMMEDIATE BLOCK: Must we block opponent's win?
        blocking_move = self.find_winning_move(opponent)
        if blocking_move:
            if self.debug:
                print(f"Found blocking move: {blocking_move}")
            return blocking_move
        
        # 3. FORK HANDLING: Check for opponent forks and our fork opportunities
        
        # 3a. Find all opponent fork moves
        opponent_fork_moves = self.find_all_fork_moves(opponent)
        
        # If there are multiple opponent fork possibilities, we're likely in trouble
        # but try our best counter-strategy
        if opponent_fork_moves:
            if self.debug:
                print(f"Opponent fork threats detected at: {opponent_fork_moves}")
            
            if self.enable_counter_threats:
                # 3b. Try to create a forcing threat that prevents opponent's fork
                counter_threat = self.find_forcing_counter_threat(opponent_fork_moves)
                if counter_threat:
                    if self.debug:
                        print(f"Found counter-threat: {counter_threat}")
                    return counter_threat
            
            # 3c. If no counter-threat is possible, block one of the fork setups
            # Choose the one that gives us the best position
            if opponent_fork_moves:
                best_block = self.find_best_fork_block(opponent_fork_moves)
                if best_block:
                    if self.debug:
                        print(f"Blocking fork with: {best_block}")
                    return best_block
        
        # 3d. Look for our own fork opportunities
        fork_move = self.find_fork_move(player)
        if fork_move:
            if self.debug:
                print(f"Creating our own fork at: {fork_move}")
            return fork_move
        
        # ------------------- STRATEGIC POSITIONING -------------------
        
        # 4. CENTER AND STRATEGIC POSITIONS
        # On small boards, take center if available
        if self.size <= 5:
            center = self.size // 2
            if board[center, center] == 0:
                if self.debug:
                    print("Taking center position")
                return (center, center)
            
            # If center is taken by opponent on 3x3, take a corner
            if self.size == 3 and board[center, center] == opponent:
                corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
                available_corners = [corner for corner in corners if board[corner[0], corner[1]] == 0]
                if available_corners:
                    corner_move = random.choice(available_corners)
                    if self.debug:
                        print(f"Taking corner: {corner_move}")
                    return corner_move
        
        # 5. THREAT CREATION: Create an attacking threat if possible
        threat_move = self.find_best_threat_move(player)
        if threat_move:
            if self.debug:
                print(f"Creating threatening position at: {threat_move}")
            return threat_move
        
        # ------------------- MINIMAX SEARCH -------------------
        
        # Start the minimax search timer
        self.start_time = time.time()
        self.time_up = False
        self.evaluation_cache = {}  # Reset cache
        
        # 6. Use iterative deepening minimax search with time limit
        if self.max_time is not None:
            max_search_depth = min(self.adaptive_depth, 10)
            
            best_move = available_moves[0]
            best_score = -infinity
            
            for depth in range(2, max_search_depth + 1):
                if self.time_up:
                    break
                
                move, score = self.minimax_root(depth)
                
                if not self.time_up:
                    best_move = move
                    best_score = score
                    
                    if self.debug:
                        print(f"Depth {depth} search found move {move} with score {score}")
        else:
            # Fixed depth search
            depth = min(self.adaptive_depth, 10)
            best_move, _ = self.minimax_root(depth)
        
        if self.debug:
            print(f"Final AI choice: {best_move}")
        
        return best_move
    
    def find_forcing_counter_threat(self, opponent_fork_moves):
        """
        Find a move that creates an immediate threat forcing the opponent to respond,
        preventing them from creating their fork.
        This is a critical counter-tactic to opponent fork setups.
        """
        player = self.game.current_player
        opponent = -player
        
        # Track the best threat move and its score
        best_move = None
        best_score = -infinity
        
        # First priority: Create a threat that forces a specific response
        # but doesn't allow any of the opponent's fork moves
        for move in self.game.get_available_moves():
            # Skip if this move is one of the opponent's fork setups
            if move in opponent_fork_moves:
                continue
            
            r, c = move
            
            # Make the move
            self.game.board[r, c] = player
            
            # See if this creates an immediate threat
            threat_lines = self.find_threat_lines(player)
            if len(threat_lines) > 0:
                # For each threat line, check if blocking it prevents all fork moves
                blocks_all_forks = True
                
                for line in threat_lines:
                    # Find the empty position in the threat line
                    empty_pos = None
                    for pos in line:
                        if self.game.board[pos] == 0:
                            empty_pos = pos
                            break
                    
                    # Check if blocking this threat allows opponent to create a fork
                    self.game.board[empty_pos] = opponent  # Simulate opponent block
                    
                    # Can opponent still create a fork after blocking?
                    can_still_fork = False
                    for fork_move in opponent_fork_moves:
                        if self.game.board[fork_move[0], fork_move[1]] == 0:  # If fork move is still available
                            # Check if it still creates a fork
                            self.game.board[fork_move[0], fork_move[1]] = opponent
                            fork_count = self.count_winning_paths(opponent)
                            self.game.board[fork_move[0], fork_move[1]] = 0
                            
                            if fork_count >= 2:
                                can_still_fork = True
                                break
                    
                    # Undo opponent's block
                    self.game.board[empty_pos] = 0
                    
                    if can_still_fork:
                        blocks_all_forks = False
                        break
                
                # If this threat forces opponent to block and prevents all forks,
                # it's an excellent counter-tactic
                if blocks_all_forks:
                    # Evaluate the resulting position
                    score = self.evaluate_board()
                    if score > best_score:
                        best_score = score
                        best_move = move
            
            # Undo our move
            self.game.board[r, c] = 0
        
        # If we found a good counter-threat, return it
        if best_move is not None:
            return best_move
        
        # Second priority: Just create any threat that doesn't allow the opponent's fork
        for move in self.game.get_available_moves():
            # Skip if this move is one of the opponent's fork setups
            if move in opponent_fork_moves:
                continue
            
            r, c = move
            
            # Make the move
            self.game.board[r, c] = player
            
            # See if this creates an immediate threat
            almost_wins = self.count_almost_wins(player)
            score = self.evaluate_board()
            
            # Undo our move
            self.game.board[r, c] = 0
            
            # If this creates a threat and gives a better position
            if almost_wins > 0 and score > best_score:
                best_score = score
                best_move = move
        
        # If we found a threat that doesn't allow opponent's fork, use it
        if best_move is not None:
            return best_move
        
        # If no good counter-threat is found, return None
        return None
    
    def find_best_fork_block(self, opponent_fork_moves):
        """Choose the best move to block opponent's fork setup"""
        player = self.game.current_player
        best_move = None
        best_score = -infinity
        
        # Try each possible fork block
        for move in opponent_fork_moves:
            r, c = move
            
            # Make the move
            self.game.board[r, c] = player
            
            # Evaluate this position
            score = self.evaluate_board()
            
            # Also check if this creates any threats for us
            our_threats = self.count_almost_wins(player)
            score += our_threats * 10  # Bonus for creating our own threats
            
            # Undo the move
            self.game.board[r, c] = 0
            
            # Keep track of best blocking move
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def find_threat_lines(self, player):
        """Find all lines where player is one move away from winning"""
        win_length = self.game.win_length
        board = self.game.board
        size = self.game.size
        threat_lines = []
        
        # Check rows
        for r in range(size):
            for c in range(size - win_length + 1):
                window = board[r, c:c+win_length]
                if np.sum(window == player) == win_length - 1 and np.sum(window == 0) == 1:
                    # Create list of positions in this line
                    line = [(r, c+i) for i in range(win_length)]
                    threat_lines.append(line)
        
        # Check columns
        for c in range(size):
            for r in range(size - win_length + 1):
                window = board[r:r+win_length, c]
                if np.sum(window == player) == win_length - 1 and np.sum(window == 0) == 1:
                    # Create list of positions in this line
                    line = [(r+i, c) for i in range(win_length)]
                    threat_lines.append(line)
        
        # Check diagonals (top-left to bottom-right)
        for r in range(size - win_length + 1):
            for c in range(size - win_length + 1):
                window = [board[r+i, c+i] for i in range(win_length)]
                if window.count(player) == win_length - 1 and window.count(0) == 1:
                    # Create list of positions in this line
                    line = [(r+i, c+i) for i in range(win_length)]
                    threat_lines.append(line)
        
        # Check diagonals (top-right to bottom-left)
        for r in range(size - win_length + 1):
            for c in range(win_length - 1, size):
                window = [board[r+i, c-i] for i in range(win_length)]
                if window.count(player) == win_length - 1 and window.count(0) == 1:
                    # Create list of positions in this line
                    line = [(r+i, c-i) for i in range(win_length)]
                    threat_lines.append(line)
        
        return threat_lines
    
    def minimax_root(self, depth):
        """Root of minimax search to find the best move"""
        available_moves = self.game.get_available_moves()
        player = self.game.current_player
        alpha = -infinity
        beta = infinity
        
        best_score = -infinity
        best_move = available_moves[0]
        
        # Try each available move
        for move in available_moves:
            r, c = move
            
            # Make the move
            self.game.board[r, c] = player
            
            # Evaluate with minimax
            score = self.minimax(depth-1, alpha, beta, False)
            
            # Undo the move
            self.game.board[r, c] = 0
            
            # Check if we're out of time
            if self.max_time and time.time() - self.start_time > self.max_time * 0.95:
                self.time_up = True
                break
            
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
            
            # Alpha-beta update
            alpha = max(alpha, best_score)
        
        return best_move, best_score
    
    def minimax(self, depth, alpha, beta, is_maximizing):
        """Minimax algorithm with alpha-beta pruning"""
        # Check for time limit
        if self.max_time and time.time() - self.start_time > self.max_time * 0.95:
            self.time_up = True
            return 0
        
        # Create a board hash for caching
        board_hash = hash(self.game.board.tobytes())
        cache_key = (board_hash, depth, is_maximizing)
        
        # Check if we already evaluated this position
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        player = self.game.current_player
        opponent = -player
        current_eval = self.evaluate_board()
        
        # Check for terminal state or max depth
        if depth == 0 or abs(current_eval) > 9000 or not self.game.get_available_moves():
            return current_eval
        
        # Maximizing player (AI)
        if is_maximizing:
            max_eval = -infinity
            for move in self.game.get_available_moves():
                r, c = move
                
                # Make move
                self.game.board[r, c] = player
                
                # Recursively evaluate
                eval = self.minimax(depth - 1, alpha, beta, False)
                
                # Undo move
                self.game.board[r, c] = 0
                
                # Update max evaluation
                max_eval = max(max_eval, eval)
                
                # Alpha-beta pruning
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
                
                # Check time limit
                if self.time_up:
                    return max_eval
            
            # Cache the result
            self.evaluation_cache[cache_key] = max_eval
            return max_eval
        
        # Minimizing player (opponent)
        else:
            min_eval = infinity
            for move in self.game.get_available_moves():
                r, c = move
                
                # Make move
                self.game.board[r, c] = opponent
                
                # Recursively evaluate
                eval = self.minimax(depth - 1, alpha, beta, True)
                
                # Undo move
                self.game.board[r, c] = 0
                
                # Update min evaluation
                min_eval = min(min_eval, eval)
                
                # Alpha-beta pruning
                beta = min(beta, eval)
                if beta <= alpha:
                    break
                
                # Check time limit
                if self.time_up:
                    return min_eval
            
            # Cache the result
            self.evaluation_cache[cache_key] = min_eval
            return min_eval
    
    def find_winning_move(self, player):
        """Find a move that would immediately win the game for the given player"""
        for move in self.game.get_available_moves():
            r, c = move
            
            # Make the move
            self.game.board[r, c] = player
            
            # Check if this wins
            temp_game = self.game.copy()
            temp_game.check_game_over()
            
            # Undo the move
            self.game.board[r, c] = 0
            
            # If this move wins, return it
            if temp_game.game_over and temp_game.winner == player:
                return move
        
        return None
    
    def find_all_fork_moves(self, player):
        """Find all moves that create a fork (two winning paths) for the given player"""
        fork_moves = []
        
        for move in self.game.get_available_moves():
            r, c = move
            
            # Make the move
            self.game.board[r, c] = player
            
            # Count the number of winning paths this creates
            winning_paths = self.count_winning_paths(player)
            
            # Undo the move
            self.game.board[r, c] = 0
            
            # If this move creates multiple winning paths, it's a fork
            if winning_paths >= 2:
                fork_moves.append(move)
        
        return fork_moves
    
    def find_fork_move(self, player):
        """Find a move that creates a fork (two winning paths) for the given player"""
        # Try to find a direct fork
        fork_moves = self.find_all_fork_moves(player)
        if fork_moves:
            return fork_moves[0]
        
        # If no immediate forks, look for good setup moves
        best_setup_move = None
        max_potential = 0
        
        for move in self.game.get_available_moves():
            r, c = move
            
            # Make the move
            self.game.board[r, c] = player
            
            # Calculate the fork potential
            potential = self.calculate_fork_potential(player)
            
            # Undo the move
            self.game.board[r, c] = 0
            
            # Update if this has better potential
            if potential > max_potential:
                max_potential = potential
                best_setup_move = move
        
        # Return a good setup move if it has significant potential
        threshold = 4 if self.size >= 5 else 3  # Higher threshold for larger boards
        if max_potential >= threshold:
            return best_setup_move
        
        return None
    
    def find_best_threat_move(self, player):
        """Find the most strategically advantageous threat move"""
        best_move = None
        best_score = -infinity
        
        for move in self.game.get_available_moves():
            r, c = move
            
            # Make the move
            self.game.board[r, c] = player
            
            # See if this creates an immediate threat
            threats = self.count_almost_wins(player)
            
            # If this creates a threat, evaluate the position
            if threats > 0:
                # Evaluate the resulting position
                score = self.evaluate_board() + threats * 5
                
                # If this is a better threat than what we've seen so far
                if score > best_score:
                    best_score = score
                    best_move = move
            
            # Undo the move
            self.game.board[r, c] = 0
        
        return best_move
    
    def calculate_fork_potential(self, player):
        """Calculate how much potential this position has for creating forks"""
        win_length = self.game.win_length
        board = self.game.board
        size = self.game.size
        opponent = -player
        
        # Count "almost winning lines" - lines with player's pieces and just one or two empty spots
        lines_with_potential = 0
        
        # Count different types of potential winning lines
        almost_win_lines = 0  # One move away from winning
        two_away_lines = 0    # Two moves away from winning
        
        # Check rows
        for r in range(size):
            for c in range(size - win_length + 1):
                window = board[r, c:c+win_length]
                player_count = np.sum(window == player)
                empty_count = np.sum(window == 0)
                opponent_count = np.sum(window == opponent)
                
                # Only consider lines without opponent pieces
                if opponent_count == 0:
                    if player_count == win_length - 1 and empty_count == 1:
                        almost_win_lines += 1
                    elif player_count == win_length - 2 and empty_count == 2:
                        two_away_lines += 1
                    elif player_count > 0 and empty_count > 0:
                        lines_with_potential += 1  # General potential
        
        # Check columns
        for c in range(size):
            for r in range(size - win_length + 1):
                window = board[r:r+win_length, c]
                player_count = np.sum(window == player)
                empty_count = np.sum(window == 0)
                opponent_count = np.sum(window == opponent)
                
                if opponent_count == 0:
                    if player_count == win_length - 1 and empty_count == 1:
                        almost_win_lines += 1
                    elif player_count == win_length - 2 and empty_count == 2:
                        two_away_lines += 1
                    elif player_count > 0 and empty_count > 0:
                        lines_with_potential += 1
        
        # Check diagonals (top-left to bottom-right)
        for r in range(size - win_length + 1):
            for c in range(size - win_length + 1):
                diag = [board[r+i, c+i] for i in range(win_length)]
                player_count = diag.count(player)
                empty_count = diag.count(0)
                opponent_count = diag.count(opponent)
                
                if opponent_count == 0:
                    if player_count == win_length - 1 and empty_count == 1:
                        almost_win_lines += 1
                    elif player_count == win_length - 2 and empty_count == 2:
                        two_away_lines += 1
                    elif player_count > 0 and empty_count > 0:
                        lines_with_potential += 1
        
        # Check diagonals (top-right to bottom-left)
        for r in range(size - win_length + 1):
            for c in range(win_length - 1, size):
                diag = [board[r+i, c-i] for i in range(win_length)]
                player_count = diag.count(player)
                empty_count = diag.count(0)
                opponent_count = diag.count(opponent)
                
                if opponent_count == 0:
                    if player_count == win_length - 1 and empty_count == 1:
                        almost_win_lines += 1
                    elif player_count == win_length - 2 and empty_count == 2:
                        two_away_lines += 1
                    elif player_count > 0 and empty_count > 0:
                        lines_with_potential += 1
        
        # Calculate fork potential score
        # Weight almost-win lines heavily as they're most important for forks
        potential_score = (almost_win_lines * 5) + (two_away_lines * 2) + lines_with_potential
        
        # Consider center control for additional strategic value
        center = size // 2
        if size % 2 == 1 and board[center, center] == player:
            potential_score += 2
        
        # For 3x3, corners are important
        if size == 3:
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            for r, c in corners:
                if board[r, c] == player:
                    potential_score += 1
        
        return potential_score
    
    def count_winning_paths(self, player):
        """Count how many winning paths (one move away from win) exist for the player"""
        win_paths = 0
        win_length = self.game.win_length
        board = self.game.board
        size = self.game.size
        
        # Check all empty positions
        for move in self.game.get_available_moves():
            r, c = move
            
            # Try the move
            board[r, c] = player
            
            # Check if this would win
            
            # Horizontal
            for col in range(max(0, c - win_length + 1), min(c + 1, size - win_length + 1)):
                window = board[r, col:col+win_length]
                if np.all(window == player):
                    win_paths += 1
                    break
            
            # Vertical
            for row in range(max(0, r - win_length + 1), min(r + 1, size - win_length + 1)):
                window = board[row:row+win_length, c]
                if np.all(window == player):
                    win_paths += 1
                    break
            
            # Diagonal (top-left to bottom-right)
            for i in range(win_length):
                if 0 <= r - i < size and 0 <= c - i < size and r - i + win_length <= size and c - i + win_length <= size:
                    diagonal = True
                    for j in range(win_length):
                        if board[r-i+j, c-i+j] != player:
                            diagonal = False
                            break
                    if diagonal:
                        win_paths += 1
                        break
            
            # Diagonal (top-right to bottom-left)
            for i in range(win_length):
                if 0 <= r - i < size and 0 <= c + i < size and r - i + win_length <= size and c + i - win_length + 1 >= 0:
                    diagonal = True
                    for j in range(win_length):
                        if board[r-i+j, c+i-j] != player:
                            diagonal = False
                            break
                    if diagonal:
                        win_paths += 1
                        break
            
            # Undo the move
            board[r, c] = 0
        
        return win_paths
    
    def evaluate_board(self):
        """Evaluate the current board position"""
        player = self.game.current_player
        opponent = -player
        board = self.game.board
        win_length = self.game.win_length
        size = self.game.size
        
        # Check for actual win/loss
        temp_game = self.game.copy()
        temp_game.check_game_over()
        
        if temp_game.game_over:
            if temp_game.winner == player:
                return 10000  # Win
            elif temp_game.winner == opponent:
                return -10000  # Loss
            else:
                return 0  # Draw
        
        # Calculate a score based on potential winning lines
        score = 0
        
        # Check rows
        for r in range(size):
            for c in range(size - win_length + 1):
                window = board[r, c:c+win_length]
                score += self.evaluate_window(window, player)
        
        # Check columns
        for c in range(size):
            for r in range(size - win_length + 1):
                window = board[r:r+win_length, c]
                score += self.evaluate_window(window, player)
        
        # Check diagonals (top-left to bottom-right)
        for r in range(size - win_length + 1):
            for c in range(size - win_length + 1):
                window = [board[r+i, c+i] for i in range(win_length)]
                score += self.evaluate_window(window, player)
        
        # Check diagonals (top-right to bottom-left)
        for r in range(size - win_length + 1):
            for c in range(win_length - 1, size):
                window = [board[r+i, c-i] for i in range(win_length)]
                score += self.evaluate_window(window, player)
        
        # Value center positions more on larger boards
        center = size // 2
        center_value = 3 if size <= 3 else 2  # Less important on larger boards
        
        if size % 2 == 1:  # Odd size has a center point
            if board[center, center] == player:
                score += center_value
            elif board[center, center] == opponent:
                score -= center_value
        else:  # Even size has 4 centers
            center_area = board[center-1:center+1, center-1:center+1]
            score += np.sum(center_area == player) * (center_value // 2)
            score -= np.sum(center_area == opponent) * (center_value // 2)
        
        # Special strategic evaluations
        
        # Count number of threats (almost-wins)
        player_threats = self.count_almost_wins(player)
        opponent_threats = self.count_almost_wins(opponent)
        
        # Threats are more valuable on larger boards
        threat_value = 5 if size <= 3 else 8
        score += player_threats * threat_value
        score -= opponent_threats * (threat_value + 5)  # Opponent threats more dangerous
        
        # Fork potential evaluation
        player_fork_potential = self.count_fork_potential(player)
        opponent_fork_potential = self.count_fork_potential(opponent)
        
        # Fork potential is critically important
        score += player_fork_potential * 25
        score -= opponent_fork_potential * 40  # Opponent forks are very dangerous
        
        return score
    
    def evaluate_window(self, window, player):
        """Evaluate a single window (row, column, or diagonal segment)"""
        opponent = -player
        win_length = self.game.win_length
        
        if not isinstance(window, np.ndarray):
            window = np.array(window)
        
        player_count = np.sum(window == player)
        opponent_count = np.sum(window == opponent)
        empty_count = np.sum(window == 0)
        
        # Can't win in this window if both players have pieces in it
        if opponent_count > 0 and player_count > 0:
            return 0
        
        score = 0
        
        # Potential win for player
        if opponent_count == 0:
            if player_count == win_length - 1 and empty_count == 1:
                score += 100  # One away from winning
            elif player_count == win_length - 2 and empty_count == 2:
                score += 10   # Two away from winning
            elif player_count > 0:
                score += player_count  # Some potential
        
        # Potential win for opponent
        if player_count == 0:
            if opponent_count == win_length - 1 and empty_count == 1:
                score -= 120  # Block opponent win (high priority)
            elif opponent_count == win_length - 2 and empty_count == 2:
                score -= 20   # Block opponent setup (increased priority)
            elif opponent_count > 0:
                score -= opponent_count  # Some concern
        
        return score
    
    def count_almost_wins(self, player):
        """Count how many lines are one move away from winning"""
        win_length = self.game.win_length
        board = self.game.board
        size = self.game.size
        almost_wins = 0
        
        # Check rows
        for r in range(size):
            for c in range(size - win_length + 1):
                window = board[r, c:c+win_length]
                if np.sum(window == player) == win_length - 1 and np.sum(window == 0) == 1:
                    almost_wins += 1
        
        # Check columns
        for c in range(size):
            for r in range(size - win_length + 1):
                window = board[r:r+win_length, c]
                if np.sum(window == player) == win_length - 1 and np.sum(window == 0) == 1:
                    almost_wins += 1
        
        # Check diagonals (top-left to bottom-right)
        for r in range(size - win_length + 1):
            for c in range(size - win_length + 1):
                window = [board[r+i, c+i] for i in range(win_length)]
                if window.count(player) == win_length - 1 and window.count(0) == 1:
                    almost_wins += 1
        
        # Check diagonals (top-right to bottom-left)
        for r in range(size - win_length + 1):
            for c in range(win_length - 1, size):
                window = [board[r+i, c-i] for i in range(win_length)]
                if window.count(player) == win_length - 1 and window.count(0) == 1:
                    almost_wins += 1
        
        return almost_wins
    
    def count_fork_potential(self, player):
        """Count how many moves could potentially create a fork"""
        fork_potential = 0
        
        for move in self.game.get_available_moves():
            r, c = move
            
            # Make the move
            self.game.board[r, c] = player
            
            # Check if this creates multiple "almost" winning lines
            almost_wins = self.count_almost_wins(player)
            
            # Undo the move
            self.game.board[r, c] = 0
            
            # If this creates multiple "almost" winning lines, it has fork potential
            if almost_wins >= 2:
                fork_potential += 1
        
        return fork_potential


def play_player_vs_player(board_size):
    game = TicTacToe(board_size)
    
    print("\nPlayer 1: X, Player 2: O")
    print(f"Win condition: Get {game.win_length} in a row")
    
    while not game.game_over:
        game.display_board()
        
        player_name = "Player 1" if game.current_player == 1 else "Player 2"
        symbol = game.player_symbols[game.current_player]
        
        try:
            move = input(f"{player_name} ({symbol}), enter move (row col): ")
            row, col = map(int, move.split())
            row -= 1  # Adjust for 0-indexing
            col -= 1
            
            if 0 <= row < game.size and 0 <= col < game.size:
                if game.make_move(row, col):
                    pass  # Move successful
                else:
                    print("Invalid move, try again!")
            else:
                print(f"Row and column must be between 1 and {game.size}!")
        except ValueError:
            print("Please enter row and column as numbers separated by space!")
    
    game.display_board()
    
    if game.winner == 0:
        print("Game ended in a draw!")
    else:
        winner = "Player 1" if game.winner == 1 else "Player 2"
        print(f"{winner} wins!")

def play_player_vs_ai(board_size, ai_time_limit):
    game = TicTacToe(board_size)
    
    # Set depth based on board size
    if board_size <= 3:
        max_depth = 9  # Full exploration for 3x3
    elif board_size <= 4:
        max_depth = 7  # Deep exploration for 4x4
    elif board_size <= 5:
        max_depth = 5  # Moderate exploration for 5x5
    else:
        max_depth = 4  # Limited depth for larger boards
    
    ai = ScalableTicTacToeAI(game, max_depth=max_depth, max_time=ai_time_limit)
    
    print("\nYou: X, AI: O")
    print(f"Win condition: Get {game.win_length} in a row")
    print(f"AI thinking time: {ai_time_limit if ai_time_limit else 'Unlimited'} seconds")
    
    while not game.game_over:
        game.display_board()
        
        if game.current_player == 1:  # Human player
            try:
                move = input("Your move (row col): ")
                row, col = map(int, move.split())
                row -= 1  # Adjust for 0-indexing
                col -= 1
                
                if 0 <= row < game.size and 0 <= col < game.size:
                    if game.make_move(row, col):
                        pass  # Move successful
                    else:
                        print("Invalid move, try again!")
                        continue
                else:
                    print(f"Row and column must be between 1 and {game.size}!")
                    continue
            except ValueError:
                print("Please enter row and column as numbers separated by space!")
                continue
        else:  # AI player
            print("AI is thinking...")
            start_time = time.time()
            move = ai.get_move()
            elapsed = time.time() - start_time
            
            if move:
                row, col = move
                print(f"AI chose: {row + 1} {col + 1} (in {elapsed:.2f} seconds)")
                game.make_move(row, col)
    
    game.display_board()
    
    if game.winner == 0:
        print("Game ended in a draw!")
    elif game.winner == 1:
        print("You win!")
    else:
        print("AI wins!")

def play_ai_vs_ai(board_size):
    game = TicTacToe(board_size)
    
    # For different AI strengths
    if board_size <= 3:
        max_depth = 9
    elif board_size <= 4:
        max_depth = 7
    else:
        max_depth = 5
    
    # Both AIs use the ScalableTicTacToeAI now
    ai1 = ScalableTicTacToeAI(game, max_depth=max_depth, max_time=3)
    ai2 = ScalableTicTacToeAI(game, max_depth=max_depth-2, max_time=2)  # Second AI is slightly weaker
    
    print("\nAI 1 (X) vs AI 2 (O)")
    print(f"Win condition: Get {game.win_length} in a row")
    
    move_counter = 0
    while not game.game_over:
        game.display_board()
        move_counter += 1
        
        if game.current_player == 1:  # AI 1 (X)
            print("AI 1 is thinking...")
            start_time = time.time()
            move = ai1.get_move()
            elapsed = time.time() - start_time
            
            if move:
                row, col = move
                print(f"AI 1 chose: {row + 1} {col + 1} (in {elapsed:.2f} seconds)")
                game.make_move(row, col)
        else:  # AI 2 (O)
            print("AI 2 is thinking...")
            start_time = time.time()
            move = ai2.get_move()
            elapsed = time.time() - start_time
            
            if move:
                row, col = move
                print(f"AI 2 chose: {row + 1} {col + 1} (in {elapsed:.2f} seconds)")
                game.make_move(row, col)
        
        # Add a delay to watch the game
        time.sleep(1)
    
    game.display_board()
    
    if game.winner == 0:
        print("Game ended in a draw!")
    elif game.winner == 1:
        print("AI 1 (X) wins!")
    else:
        print("AI 2 (O) wins!")

def main():
    print("Welcome to Dynamic Tic Tac Toe!")
    
    # Get board size
    while True:
        try:
            board_size = int(input("Enter board size (3 for 3x3, 4 for 4x4, etc.): "))
            if board_size >= 3:
                break
            print("Board size must be at least 3!")
        except ValueError:
            print("Please enter a valid number!")
    
    # Get game mode
    print("\nGame Modes:")
    print("1. Player vs Player")
    print("2. Player vs AI")
    print("3. AI vs AI")
    
    while True:
        try:
            mode = int(input("Select game mode (1-3): "))
            if 1 <= mode <= 3:
                break
            print("Please enter a number between 1 and 3!")
        except ValueError:
            print("Please enter a valid number!")
    
    # Play the selected game mode
    if mode == 1:
        play_player_vs_player(board_size)
    elif mode == 2:
        # Get AI thinking time
        print("\nAI Thinking Time:")
        print("1. Quick (1 second)")
        print("2. Medium (5 seconds)")
        print("3. Slow (10 seconds)")
        print("4. Very Slow (20 seconds)")
        print("5. Unlimited")
        
        while True:
            try:
                time_option = int(input("Select AI thinking time (1-5): "))
                if 1 <= time_option <= 5:
                    break
                print("Please enter a number between 1 and 5!")
            except ValueError:
                print("Please enter a valid number!")
        
        ai_time_limits = {1: 1, 2: 5, 3: 10, 4: 20, 5: None}
        ai_time_limit = ai_time_limits[time_option]
        
        play_player_vs_ai(board_size, ai_time_limit)
    else:  # mode == 3
        play_ai_vs_ai(board_size)
    
    print("Thanks for playing!")

if __name__ == "__main__":
    main()
