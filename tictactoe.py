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

class TicTacToeAI:
    def __init__(self, game, max_depth=None, max_time=None, find_best=True):
        self.game = game
        self.max_depth = max_depth if max_depth is not None else infinity
        self.max_time = max_time
        self.start_time = None
        self.find_best = find_best  # If True, find best move; if False, find worst move
        self.time_up = False
    
    def get_best_move(self):
        available_moves = self.game.get_available_moves()
        
        if not available_moves:
            return None
        
        # If it's the first move on a large board, choose a random move near center to save time
        if len(available_moves) > self.game.size ** 2 - 2 and self.game.size >= 4:
            center = self.game.size // 2
            center_moves = [(r, c) for r, c in available_moves 
                           if abs(r - center) <= 1 and abs(c - center) <= 1]
            if center_moves:
                return random.choice(center_moves)
            return random.choice(available_moves)
        
        self.start_time = time.time()
        self.time_up = False
        
        best_score = -infinity if self.find_best else infinity
        best_move = available_moves[0]
        player = self.game.current_player
        
        # Iterative deepening when time limit is specified
        if self.max_time is not None:
            for depth in range(1, min(self.max_depth, 10) + 1):
                if self.time_up:
                    break
                
                current_best_score = -infinity if self.find_best else infinity
                current_best_move = None
                
                for row, col in available_moves:
                    # Make move
                    self.game.board[row, col] = player
                    
                    # Calculate score
                    if self.find_best:
                        score = self.minimax(depth - 1, -infinity, infinity, False, player)
                        if score > current_best_score:
                            current_best_score = score
                            current_best_move = (row, col)
                    else:
                        score = self.minimax(depth - 1, -infinity, infinity, True, player)
                        if score < current_best_score:
                            current_best_score = score
                            current_best_move = (row, col)
                    
                    # Undo move
                    self.game.board[row, col] = 0
                    
                    # Check time limit
                    if self.max_time is not None and time.time() - self.start_time > self.max_time * 0.9:
                        self.time_up = True
                        break
                
                if current_best_move is not None and not self.time_up:
                    best_move = current_best_move
                    best_score = current_best_score
        else:
            # No time limit, use max_depth
            for row, col in available_moves:
                # Make move
                self.game.board[row, col] = player
                
                # Calculate score
                if self.find_best:
                    score = self.minimax(self.max_depth - 1, -infinity, infinity, False, player)
                    if score > best_score:
                        best_score = score
                        best_move = (row, col)
                else:
                    score = self.minimax(self.max_depth - 1, -infinity, infinity, True, player)
                    if score < best_score:
                        best_score = score
                        best_move = (row, col)
                
                # Undo move
                self.game.board[row, col] = 0
        
        return best_move
    
    def minimax(self, depth, alpha, beta, is_maximizing, player):
        # Check time limit
        if self.max_time is not None and time.time() - self.start_time > self.max_time * 0.9:
            self.time_up = True
            return 0
        
        # Check terminal conditions
        game_result = self.evaluate_terminal(self.game.board, self.game.win_length)
        if game_result != 0:
            return game_result * (depth + 1) * player  # Adjust by depth for faster wins
        
        # Check if board is full (draw)
        if not np.any(self.game.board == 0):
            return 0
        
        # Check if reached max depth
        if depth == 0:
            return self.evaluate_position(self.game.board, player)
        
        available_moves = self.get_available_moves()
        
        if is_maximizing:
            best_score = -infinity
            for row, col in available_moves:
                # Make move
                self.game.board[row, col] = player
                
                # Recursively calculate score
                score = self.minimax(depth - 1, alpha, beta, False, player)
                
                # Undo move
                self.game.board[row, col] = 0
                
                if self.time_up:
                    return best_score
                
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            
            return best_score
        else:
            best_score = infinity
            for row, col in available_moves:
                # Make move
                self.game.board[row, col] = -player
                
                # Recursively calculate score
                score = self.minimax(depth - 1, alpha, beta, True, player)
                
                # Undo move
                self.game.board[row, col] = 0
                
                if self.time_up:
                    return best_score
                
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            
            return best_score
    
    def evaluate_terminal(self, board, win_length):
        size = self.game.size
        
        # Check rows
        for r in range(size):
            for c in range(size - win_length + 1):
                if board[r, c] != 0:
                    if np.all(board[r, c:c+win_length] == board[r, c]):
                        return board[r, c]  # Someone won
        
        # Check columns
        for c in range(size):
            for r in range(size - win_length + 1):
                if board[r, c] != 0:
                    if np.all(board[r:r+win_length, c] == board[r, c]):
                        return board[r, c]  # Someone won
        
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
                        return board[r, c]  # Someone won
        
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
                        return board[r, c]  # Someone won
        
        return 0  # No winner
    
    def get_available_moves(self):
        moves = []
        for r in range(self.game.size):
            for c in range(self.game.size):
                if self.game.board[r, c] == 0:
                    moves.append((r, c))
        return moves
    
    def evaluate_position(self, board, player):
        # Heuristic evaluation for non-terminal positions
        score = 0
        size = self.game.size
        win_length = self.game.win_length
        
        # Check for potential winning lines
        
        # Rows
        for r in range(size):
            for c in range(size - win_length + 1):
                window = board[r, c:c+win_length]
                score += self.evaluate_window(window, player)
        
        # Columns
        for c in range(size):
            for r in range(size - win_length + 1):
                window = board[r:r+win_length, c]
                score += self.evaluate_window(window, player)
        
        # Diagonals (top-left to bottom-right)
        for r in range(size - win_length + 1):
            for c in range(size - win_length + 1):
                window = np.array([board[r+i, c+i] for i in range(win_length)])
                score += self.evaluate_window(window, player)
        
        # Diagonals (top-right to bottom-left)
        for r in range(size - win_length + 1):
            for c in range(win_length - 1, size):
                window = np.array([board[r+i, c-i] for i in range(win_length)])
                score += self.evaluate_window(window, player)
        
        # Favor center positions
        center = size // 2
        if size % 2 == 1:  # Odd size board has a single center
            if board[center, center] == player:
                score += 3
            elif board[center, center] == -player:
                score -= 3
        else:  # Even size board has 4 centers
            center_window = board[center-1:center+1, center-1:center+1]
            score += np.sum(center_window == player) * 2
            score -= np.sum(center_window == -player) * 2
        
        return score
    
    def evaluate_window(self, window, player):
        # Evaluate a single window (row, column, or diagonal segment)
        score = 0
        opponent = -player
        
        player_count = np.sum(window == player)
        opponent_count = np.sum(window == opponent)
        empty_count = np.sum(window == 0)
        
        # Scoring based on piece counts in window
        if player_count == self.game.win_length:
            score += 100  # Winning window
        elif player_count == self.game.win_length - 1 and empty_count == 1:
            score += 10  # One move away from winning
        elif player_count == self.game.win_length - 2 and empty_count == 2:
            score += 1  # Two moves away from winning
        
        if opponent_count == self.game.win_length - 1 and empty_count == 1:
            score -= 8  # Block opponent's winning move
        
        return score

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
    
    # Calculate appropriate max_depth based on board size
    if board_size == 3:
        max_depth = 9  # Full depth for 3x3
    elif board_size == 4:
        max_depth = 6  # Reasonable depth for 4x4
    elif board_size == 5:
        max_depth = 4  # Limited depth for 5x5
    else:
        max_depth = 3  # Very limited depth for larger boards
    
    ai = TicTacToeAI(game, max_depth=max_depth, max_time=ai_time_limit)
    
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
            move = ai.get_best_move()
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
    
    # For larger boards, use smaller depths to keep the game moving
    if board_size <= 3:
        max_depth = 9
    elif board_size <= 4:
        max_depth = 6
    else:
        max_depth = 4
    
    # Smart AI (maximizing)
    smart_ai = TicTacToeAI(game, max_depth=max_depth, max_time=2, find_best=True)
    # Dumb AI (minimizing)
    dumb_ai = TicTacToeAI(game, max_depth=max_depth, max_time=2, find_best=False)
    
    print("\nSmart AI (X) vs Dumb AI (O)")
    print(f"Win condition: Get {game.win_length} in a row")
    
    move_counter = 0
    while not game.game_over:
        game.display_board()
        move_counter += 1
        
        if game.current_player == 1:  # Smart AI (X)
            print("Smart AI is thinking...")
            start_time = time.time()
            move = smart_ai.get_best_move()
            elapsed = time.time() - start_time
            
            if move:
                row, col = move
                print(f"Smart AI chose: {row + 1} {col + 1} (in {elapsed:.2f} seconds)")
                game.make_move(row, col)
        else:  # Dumb AI (O)
            print("Dumb AI is thinking...")
            start_time = time.time()
            move = dumb_ai.get_best_move()
            elapsed = time.time() - start_time
            
            if move:
                row, col = move
                print(f"Dumb AI chose: {row + 1} {col + 1} (in {elapsed:.2f} seconds)")
                game.make_move(row, col)
        
        # Add a delay to watch the game
        time.sleep(1)
    
    game.display_board()
    
    if game.winner == 0:
        print("Game ended in a draw!")
    elif game.winner == 1:
        print("Smart AI (X) wins!")
    else:
        print("Dumb AI (O) wins!")

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
    print("3. AI vs AI (Smart vs Dumb)")
    
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
