#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#define BOARD_FILE "board.txt"
#define N 9
#define R 81
#define C 90
#define S 99
#define BSIZE 108
#define ITERATIONS 20
#define BLOCKS 256
#define THREADS 512
typedef unsigned int Uint;
#define SET_BIT(a,b) a |= ((Uint)1 << b);
#define CLEAR_BIT(a, b) a &= ~((Uint)1 << b);
#define CHECK_BIT(a, b) (bool)((a >> b) & (Uint)1)
#define GET_ROW(i) (R + i / N);
#define GET_COLUMN(i) (int)(C + i % N);
#define GET_SUBBOARD(i) (int)(S + ((i / N) / 3) * 3 + ((i % N) / 3));

Uint* solveSudoku(Uint* board);
void printBoard(Uint* board, bool printAdditional);
Uint* initializeBoard(Uint* board);
void initializeRows(Uint* board);
void initializeColumns(Uint* board);
void initializeSubboards(Uint* board);
Uint* loadBoard();

__global__ void cudaBFS(Uint* oldBoard, Uint* newBoard, int boardsCount, int *lastBoard, int* empties, int isFinal)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while(idx < boardsCount)
    {
        int boardBegin = idx * BSIZE;
        int i = boardBegin; 
        while(i < boardBegin + N * N)
        {
            if(oldBoard[i] == 0)
                break;
            i++;
        }
        if (i == boardBegin + N * N)
            return;

        int r = boardBegin + GET_ROW((i - boardBegin));
        int c = boardBegin + GET_COLUMN((i - boardBegin));
        int s = boardBegin + GET_SUBBOARD((i - boardBegin));
        for (int number = 1; number <= N; number++)
        {
            if (CHECK_BIT(oldBoard[r], (number - 1)) && CHECK_BIT(oldBoard[c], (number - 1)) && CHECK_BIT(oldBoard[s], (number - 1)))
            {
                int curBoard = atomicAdd(lastBoard, 1);
                int newBegin = curBoard * BSIZE;
                int emptiesIdx = curBoard * N * N;
                for (int j = 0; j < BSIZE; j++)
                {
                    newBoard[newBegin + j] = oldBoard[boardBegin + j];
                    if (isFinal == 1 && oldBoard[boardBegin + j] == 0 && boardBegin + j != newBegin + i - boardBegin)
                    {
                        empties[emptiesIdx] = j;
                        emptiesIdx++;
                    }
                }
                newBoard[newBegin + i - boardBegin] = number;
                CLEAR_BIT(newBoard[newBegin + r - boardBegin], (number - 1));
                CLEAR_BIT(newBoard[newBegin + c - boardBegin], (number - 1));
                CLEAR_BIT(newBoard[newBegin + s - boardBegin], (number - 1));
            }
        }
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void cudaDFS(Uint* board, int boardsCount, int* empties, int* outStatus, Uint* solution)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while ((*outStatus) == 0 && idx < boardsCount)
    {
        int boardBegin = idx * BSIZE;
        int emptiesBegin = idx * N * N;
        int emptiesIdx = emptiesBegin;

        while (emptiesIdx >= emptiesBegin && empties[emptiesIdx] >= 0)
        {
            int boardIdx = boardBegin + empties[emptiesIdx];
            int r = boardBegin + GET_ROW((boardIdx - boardBegin));
            int c = boardBegin + GET_COLUMN((boardIdx - boardBegin));
            int s = boardBegin + GET_SUBBOARD((boardIdx - boardBegin));

            board[boardIdx]++;
            Uint number = board[boardIdx];
            if (CHECK_BIT(board[r], (number - 1)) && CHECK_BIT(board[c], (number - 1)) && CHECK_BIT(board[s], (number - 1)))
            {
                CLEAR_BIT(board[r], (number - 1));
                CLEAR_BIT(board[c], (number - 1));
                CLEAR_BIT(board[s], (number - 1));
                emptiesIdx++;
            }
            else
            {
                if (board[boardIdx] >= 9)
                {
                    board[emptiesIdx] = 0;
                    emptiesIdx--;
                    boardIdx = boardBegin + empties[emptiesIdx];
                    r = boardBegin + GET_ROW((boardIdx - boardBegin));
                    c = boardBegin + GET_COLUMN((boardIdx - boardBegin));
                    s = boardBegin + GET_SUBBOARD((boardIdx - boardBegin));
                    number = board[boardIdx];

                    SET_BIT(board[r], (number - 1));
                    SET_BIT(board[c], (number - 1));
                    SET_BIT(board[s], (number - 1));
                }
            }
        }

        if (emptiesIdx < 0)
        {
            *outStatus = 1;
            for (int i = 0; i < N * N; i++)
            {
                solution[i] = board[boardBegin + i];
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

int main()
{
    /*Uint board[]{
        0, 4, 0, 0, 0, 0, 1, 7, 9,
        0, 0, 2, 0, 0, 8, 0, 5, 4,
        0, 0, 6, 0, 0, 5, 0, 0, 8,
        0, 8, 0, 0, 7, 0, 9, 1, 0,
        0, 5, 0, 0, 9, 0, 0, 3, 0,
        0, 1, 9, 0, 6, 0, 0, 4, 0,
        3, 0, 0, 4, 0, 0, 7, 0, 0,
        5, 7, 0, 1, 0, 0, 2, 0, 0,
        9, 2, 8, 0, 0, 0, 0, 6, 0 };*/
    Uint* board = loadBoard();
    solveSudoku(board);
    return 0;
}

Uint* solveSudoku(Uint *inBoard)
{
    Uint* board = initializeBoard(inBoard);
    printBoard(board, true);

    const int boardMem = pow(2, 26);
    Uint* board1, *board2;
    int* lastBoard, *empties, *emptiesCount;

    cudaMalloc(&board1, boardMem * sizeof(Uint));
    cudaMalloc(&board2, boardMem * sizeof(Uint));
    cudaMalloc(&empties, boardMem * sizeof(Uint));
    cudaMalloc(&emptiesCount, (boardMem / BSIZE) * sizeof(Uint));
    cudaMalloc(&lastBoard, sizeof(int));

    cudaMemset(board1, 0, boardMem * sizeof(Uint));
    cudaMemset(board2, 0, boardMem * sizeof(Uint));
    cudaMemset(empties, -1, boardMem * sizeof(Uint));
    //cudaMemset(emptiesCount, 0, (boardMem / BSIZE) * sizeof(Uint));

    cudaMemcpy(board1, board, BSIZE * sizeof(Uint), cudaMemcpyHostToDevice);
    int boardsCount = 1;
    int isFinal = 0;
    
    for (int it = 0; it < ITERATIONS; it++)
    {
        cudaMemset(lastBoard, 0, sizeof(int));
        
        if (it == ITERATIONS - 1)
            isFinal = 1;

        if(it % 2 == 0)
            cudaBFS<<< BLOCKS, THREADS >>>(board1, board2, boardsCount, lastBoard, empties, isFinal);
        else
            cudaBFS<<< BLOCKS, THREADS >>>(board2, board1, boardsCount, lastBoard, empties, isFinal);

        cudaMemcpy(&boardsCount, lastBoard, sizeof(int), cudaMemcpyDeviceToHost);
        
         printf("boards count after it %d: %d\n", it, boardsCount);
    }

    Uint* bfsBoard = (ITERATIONS % 2 == 0)? board1 : board2;

    cudaMemcpy(board, bfsBoard, BSIZE * sizeof(Uint), cudaMemcpyDeviceToHost);
    printBoard(board, true);

    int *outStatus;
    Uint* solution;

    cudaMalloc(&outStatus, sizeof(int));
    cudaMalloc(&solution, N * N * sizeof(Uint));
    
    cudaMemset(outStatus, 0, sizeof(int));
    cudaMemset(solution, 0, N * N * sizeof(Uint));

    cudaDFS<<< BLOCKS, THREADS >>>(bfsBoard, boardsCount, empties, outStatus, solution);

    cudaMemcpy(board, solution, N * N * sizeof(Uint), cudaMemcpyDeviceToHost);
    printBoard(board, false);

    cudaFree(board1);
    cudaFree(board2);
    cudaFree(lastBoard);
    cudaFree(empties);
    cudaFree(outStatus);
    cudaFree(solution);
    return board;
}

void printBoard(Uint* board, bool printAdditional)
{
    std::cout << "\n\n";
    for (int i = 0; i < N; i++)
    {
        if (i == 3 || i == 6)
            std::cout << "-------------------\n";
        for (int j = 0; j < N; j++)
        {
            if (j == 3 || j == 6)
                std::cout << "|";
            std::cout << board[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    if (!printAdditional)
        return;
    std::cout << "\n" << "rows: " << "\n";
    for (int i = 0; i < N; i++)
    {
        std::cout << board[R + i] << " ";
    }
    std::cout << "\n" << "columns: " << "\n";
    for (int i = 0; i < N; i++)
    {
        std::cout << board[C + i] << " ";
    }
    std::cout << "\n" << "subboards: " << "\n";
    for (int i = 0; i < N; i++)
    {
        std::cout << board[S + i] << " ";
    }
}

Uint* initializeBoard(Uint* inBoard)
{
    Uint *board = (Uint*)malloc(BSIZE * sizeof(Uint));
    std::copy(inBoard, inBoard + N * N, board);
    initializeRows(board);
    initializeColumns(board);
    initializeSubboards(board);

    return board;
}
void initializeRows(Uint* board)
{
    for (int r = 0; r < N; r++)
    {
        board[R + r] = (Uint)pow(2, N) - 1;
        for (int i = 0; i < N; i++)
        {
            if (board[r * N + i] > 0)
                CLEAR_BIT(board[R + r], (board[r * N + i] - 1));
        }
    }
}

void initializeColumns(Uint* board)
{
    for (int c = 0; c < N; c++)
    {
        board[C + c] = (Uint)pow(2, N) - 1;
        for (int i = 0; i < N; i++)
        {
            if (board[i * N + c] > 0)
                CLEAR_BIT(board[C + c], (board[i * N + c] - 1));
        }
    }
}

void initializeSubboards(Uint* board)
{
    for (int s = 0; s < N; s++)
    {
        board[S + s] = (Uint)pow(2, N) - 1;
        int start = ((s / 3) * 3) * N + ((s % 3) * 3);
        for (int i = 0; i < N; i++)
        {
            int r = i / 3;
            int c = i % 3;
            if (board[start + r * N + c] > 0)
                CLEAR_BIT(board[S + s], (board[start + r * N + c] - 1));
        }
    }
}

Uint* loadBoard()
{
    FILE* boardFile = fopen(BOARD_FILE, "r");
    Uint* board = (Uint*)malloc(N * N * sizeof(Uint));
    char c;

    if (boardFile == NULL)
    {
        printf("boardFile error\n");
        return board;
    }

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            if (!fscanf(boardFile, "%c\n", &c)) 
            {
                printf("boardFile error\n");
                return board;
            }

            if (c >= '1' && c <= '9') {
                board[i * N + j] = (Uint)(c - '0');
            }
            else {
                board[i * N + j] = 0;
            }
        }
    }
    return board;
}