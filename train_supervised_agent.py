#type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_PATH = 'othello_dataset.csv'
MODEL_PATH = 'supervised_model.pth'
BOARD_SIZE = 8  # 8x8 Othello
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

class OthelloNet(nn.Module):
    def __init__(self, board_size=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * board_size * board_size, 1024), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, board_size * board_size)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def move_str_to_coords(move_str):
    col = ord(move_str[0]) - ord('a')
    row = int(move_str[1]) - 1
    return (row, col)

def encode_board(board, player):
    arr = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    arr[0] = (board == player)
    arr[1] = (board == 3 - player)
    return arr

def apply_move(board, move, player):
    from helpers import execute_move
    board = board.copy()
    execute_move(board, move, player)
    return board

def augment_board(board, move):
    k = np.random.randint(4)
    board = np.rot90(board, k)
    if k > 0:
        move = (move[0], move[1])
        for _ in range(k):
            move = (move[1], BOARD_SIZE - 1 - move[0])
    
    if np.random.random() < 0.5:
        board = np.fliplr(board)
        move = (move[0], BOARD_SIZE - 1 - move[1])
    
    if np.random.random() < 0.5:
        board = np.flipud(board)
        move = (BOARD_SIZE - 1 - move[0], move[1])
    
    return board, move

def extract_training_data():
    df = pd.read_csv(DATA_PATH)
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        moves = row['game_moves'].split('h')
        moves = [row['game_moves'][i:i+2] for i in range(0, len(row['game_moves']), 2)]
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        mid = BOARD_SIZE // 2
        board[mid-1][mid-1] = 2
        board[mid-1][mid] = 1
        board[mid][mid-1] = 1
        board[mid][mid] = 2
        player = 1
        for move_str in moves:
            if len(move_str) != 2:
                continue
            move = move_str_to_coords(move_str)
            X.append(encode_board(board, player))
            y.append(move[0] * BOARD_SIZE + move[1])
            
            for _ in range(3):
                aug_board, aug_move = augment_board(board.copy(), move)
                X.append(encode_board(aug_board, player))
                y.append(aug_move[0] * BOARD_SIZE + aug_move[1])
            
            board = apply_move(board, move, player)
            player = 3 - player
    X = np.stack(X)
    y = np.array(y)
    return X, y

def train():
    X, y = extract_training_data()
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = OthelloNet(BOARD_SIZE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        
        avg_loss = total_loss / len(dataset)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'New best model saved with loss: {best_loss:.4f}')
    
    print(f'Training completed. Best model saved to {MODEL_PATH}')

if __name__ == '__main__':
    train() 