#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#define N 9
#define R 81
#define C 90
#define S 99
#define BSIZE 108
typedef unsigned int Uint;
#define SET_BIT(a, b) a &= ~((Uint)1 << b);

Uint* solveSudoku(Uint* board);
void printBoard(Uint* board);
Uint* initializeBoard(Uint* board);
void initializeRows(Uint* board);
void initializeColumns(Uint* board);
void initializeSubboards(Uint* board);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    Uint board[]{
        0, 4, 0, 0, 0, 0, 1, 7, 9,
        0, 0, 2, 0, 0, 8, 0, 5, 4,
        0, 0, 6, 0, 0, 5, 0, 0, 8,
        0, 8, 0, 0, 7, 0, 9, 1, 0,
        0, 5, 0, 0, 9, 0, 0, 3, 0,
        0, 1, 9, 0, 6, 0, 0, 4, 0,
        3, 0, 0, 4, 0, 0, 7, 0, 0,
        5, 7, 0, 1, 0, 0, 2, 0, 0,
        9, 2, 8, 0, 0, 0, 0, 6, 0 };
    //printBoard(board);
    solveSudoku(board);
    return 0;
}

Uint* solveSudoku(Uint *inBoard)
{
    Uint* board = initializeBoard(inBoard);
    return board;
}

void printBoard(Uint* board)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << board[i * N + j] << " ";
        }
        std::cout << "\n";
    }
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
    Uint board[BSIZE] = {0};
    std::copy(inBoard, inBoard + N * N, std::begin(board));
    initializeRows(board);
    initializeColumns(board);
    initializeSubboards(board);
    printBoard(board);
    
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
                SET_BIT(board[R + r], board[r * N + i] - 1);
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
                SET_BIT(board[C + c], board[i * N + c] - 1);
                //board[C + c] = board[C + c] | ((Uint)1 << board[i * N + c]);
        }
    }
}

void initializeSubboards(Uint* board)
{
    for (int s = 0; s < N; s++)
    {
        board[S + s] = (Uint)pow(2, N - 1);
        int start = ((s / 3) * 3) * 3 * N + ((s % 3) * 3) * 3;
        for (int i = 0; i < N; i++)
        {
            int r = i / 3;
            int c = i % 3;
            if (board[start + r * N + c] > 0)
                board[S + s] = board[S + s] | ((Uint)1 << board[start + r * N + c]);
        }
    }
}