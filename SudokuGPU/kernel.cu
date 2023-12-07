#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include "sudokuCPU.h"

using namespace std::chrono;

// Nazwa pliku z wejściowym sudoku
#define BOARD_FILE "emptyBoard.txt"
#define BLOCKS 256
#define THREADS 1024
// Rozmiar pamięci na tablice - potęga 2
#define MEMSIZE 28
// Ilość wykonywanych iteracji BFS
#define ITERATIONS 18
// Stałe pomocnicze w reprezentacji sudoku
#define N 9
#define R 81
#define C 90
#define S 99
#define BSIZE 108
typedef unsigned short Uint;
// Makra do ustawiania masek bitowych dla wierszy, kolumn, kwadratów
#define SET_BIT(a,b) a |= ((Uint)1 << b);
#define CLEAR_BIT(a, b) a &= ~((Uint)1 << b);
#define CHECK_BIT(a, b) (bool)((a >> b) & (Uint)1)
#define GET_ROW(i) (R + i / N);
#define GET_COLUMN(i) (int)(C + i % N);
#define GET_SUBBOARD(i) (int)(S + ((i / N) / 3) * 3 + ((i % N) / 3));

void testCPU(char* data);
Uint* solveSudoku(Uint* board, int iterations);
void printCPUBoard(char* solution);
void printBoard(Uint* board, bool printAdditional);
Uint* initializeBoard(Uint* board);
void initializeRows(Uint* board);
void initializeColumns(Uint* board);
void initializeSubboards(Uint* board);
Uint* loadBoard(std::string fileName, char* cBoard);
bool checkSolution(Uint* solution, Uint* board);
__global__ void cudaDoBFS(Uint* oldBoard, Uint* newBoard, int boardsCount, int* lastBoard, int* empties, int isFinal);
__global__ void cudaDoDFS(Uint* board, int boardsCount, int* empties, int* outStatus, Uint* solution);

int main(int argc, char* argv[])
{
    std::string fileName = BOARD_FILE;
    int iterations = ITERATIONS;
    if (argc > 1)
        fileName = argv[1];
    if (argc > 2)
        iterations = std::stoi(argv[2]);
    // Wczytanie tablicy
    char data[N * N];
    Uint* board = loadBoard(fileName, data);
    
    std::cout << "Input:";
    printBoard(board, false);

    // Pomiar Prędkości rozwiązania CPU
    testCPU(data);

    auto clockStart = high_resolution_clock::now();

    // Rozwiąznaie sudoku na GPU
    Uint* solution = solveSudoku(board, iterations);

    auto clockStop = high_resolution_clock::now();
    std::cout << "\nFull Time:    " << std::setw(7) << 0.001 * duration_cast<microseconds>(clockStop - clockStart).count() << " milisec\n";

    std::cout << "\nSolution:";

    printBoard(solution, false);

    // Sprawdzenie poprawności rozwiązania
    if (checkSolution(solution, board))
        std::cout << "\nCORRECT SOLUTION!!";
    else
        std::cout << "\nINCORRECT SOLUTION!!";

    std::getchar();
    return 0;
}

void testCPU(char *data)
{
    char solution[81];
    auto clockStart = high_resolution_clock::now();
    int res = sudokuCPU(data, solution);
    auto clockStop = high_resolution_clock::now();
    std::cout << "\nCPU Solution:    " << std::setw(7) << 0.001 * duration_cast<microseconds>(clockStop - clockStart).count() << " milisec\n";
    std::cout << "CPU res: " << res;
    printCPUBoard(solution);
}

Uint* solveSudoku(Uint *inBoard, int iterations)
{
    cudaError_t cudaStatus;
    std::chrono::steady_clock::time_point clockStart, clockStop;
    clockStart = high_resolution_clock::now();

    // Dodanie dla sudoku flag reprezentujących,
    // które liczby można wpisać w dany wiersz, kolumnę, kwadrat
    Uint* board = initializeBoard(inBoard);

    // Maksymalny rozmiar tablicy
    const int boardMem = pow(2, MEMSIZE);
    
    // Dwie tablice zawierające obecny stan sudoku
    Uint *board1, *board2;
    
    // Tablica wyjściowa na rozwiązanie
    Uint *solution;

    // Indeks ostatniego sudoku w tablicy wejściowej BFS
    int* lastBoard;

    // Tablica przechowująca kolejne indeksy pustych pól dla danego sudoku
    // jeśli skończyły się puste pola przyjmuje wartości -1
    int *empties;

    // Status rozwiązania DFS, równy 1 jeśli znaleziono rozwiązanie
    int *outStatus;

    cudaMalloc(&board1, boardMem * sizeof(Uint));
    cudaMalloc(&board2, boardMem * sizeof(Uint));
    cudaMalloc(&empties, boardMem * sizeof(Uint));
    cudaMalloc(&lastBoard, sizeof(int));
    cudaMalloc(&outStatus, sizeof(int));
    cudaMalloc(&solution, N * N * sizeof(Uint));

    cudaMemset(board1, 0, boardMem * sizeof(Uint));
    cudaMemset(board2, 0, boardMem * sizeof(Uint));
    cudaMemset(empties, -1, boardMem * sizeof(Uint));
    cudaMemset(outStatus, 0, sizeof(int));
    cudaMemset(solution, 0, N * N * sizeof(Uint));

    // Skopiowanie zainicjowanego sudoku na początek tablicy wejściowej BFS
    cudaMemcpy(board1, board, BSIZE * sizeof(Uint), cudaMemcpyHostToDevice);

    // Liczba sudoku w tablicy wejściowej do BFS
    int boardsCount = 1;

    // Flaga, oznaczająca czy dana iteracja BFS jest ostatnią (uzupełniana jest wtedy tablica empties)
    int isFinal = 0;

    clockStop = high_resolution_clock::now();
    std::cout << "\nMemory allocation:    " << std::setw(7) << 0.001 * duration_cast<microseconds>(clockStop - clockStart).count() << " milisec\n\n";
    clockStart = high_resolution_clock::now();

    for (int it = 0; it < iterations; it++)
    {
        cudaMemset(lastBoard, 0, sizeof(int));
        
        if (it == iterations - 1)
            isFinal = 1;

        if(it % 2 == 0)
            cudaDoBFS<<< BLOCKS, THREADS >>>(board1, board2, boardsCount, lastBoard, empties, isFinal);
        else
            cudaDoBFS<<< BLOCKS, THREADS >>>(board2, board1, boardsCount, lastBoard, empties, isFinal);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "error launching BFS: %s\n", cudaGetErrorString(cudaStatus));

        cudaMemcpy(&boardsCount, lastBoard, sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("boards count after it %d: %d\n", it, boardsCount);

        // Liczba sudoku w tablicy przekroczyła zadaną pamięć
        if (boardsCount > (boardMem / BSIZE))
        {
            std::cout << "\n memory overflow, bfs needs more memory, exiting... \n";
            return inBoard;
        }
        cudaDeviceSynchronize();
    }

    Uint* bfsBoard = (iterations % 2 == 0)? board1 : board2;

    clockStop = high_resolution_clock::now();
    std::cout << "\nBFS:    " << std::setw(7) << 0.001 * duration_cast<microseconds>(clockStop - clockStart).count() << " milisec\n";

    clockStart = high_resolution_clock::now();

    cudaDoDFS<<< BLOCKS, THREADS >>>(bfsBoard, boardsCount, empties, outStatus, solution);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "error launching DFS: %s\n", cudaGetErrorString(cudaStatus));

    cudaDeviceSynchronize();
    clockStop = high_resolution_clock::now();
    std::cout << "\nDFS:    " << std::setw(7) << 0.001 * duration_cast<microseconds>(clockStop - clockStart).count() << " milisec\n";

    cudaStatus = cudaMemcpy(board, solution, N * N * sizeof(Uint), cudaMemcpyDeviceToHost);

    if(cudaStatus != cudaSuccess)
        fprintf(stderr, "error getting solution: %s\n", cudaGetErrorString(cudaStatus));

    cudaFree(board1);
    cudaFree(board2);
    cudaFree(lastBoard);
    cudaFree(empties);
    cudaFree(outStatus);
    cudaFree(solution);

    return board;
}

void printCPUBoard(char* solution)
{
    Uint uSol[N * N];
    for (int i = 0; i < N * N; i++)
    {
        uSol[i] = (Uint)(solution[i] - '0');
    }
    printBoard(uSol, false);
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

    // Uzupełnij maski dla wierszy
    initializeRows(board);
    
    // Uzupełnij maski dla kolumn
    initializeColumns(board);
    
    // Uzupełnij maski dla kwadratów
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

Uint* loadBoard(std::string fileName, char *cBoard)
{
    FILE* boardFile = fopen(fileName.c_str(), "r");

    Uint* board = (Uint*)malloc(N * N * sizeof(Uint));
    char c;

    if (!boardFile)
    {
        std::cout << "File: " << fileName << " does not exist!";
        return board;
    }

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
            cBoard[i * N + j] = c;
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

void clearChecked(bool* checked)
{
    for (int i = 0; i < N; i++)
    {
        checked[i] = false;
    }
}

bool checkSolution(Uint* solution, Uint* board)
{
    bool isValid = true;
    for (int i = 0; i < N * N; i++)
    {
        if (board[i] > 0 && solution[i] != board[i])
            isValid = false;
        if (solution[i] == 0)
            return false;
    }
    bool checked[N];
    for (int r = 0; r < N; r++)
    {
        clearChecked(checked);
        for (int i = 0; i < N; i++)
        {
            if (checked[solution[r * N + i] - 1] == true)
                isValid = false;
            checked[solution[r * N + i] - 1] = true;
        }
    }
    for (int c = 0; c < N; c++)
    {
        clearChecked(checked);
        for (int i = 0; i < N; i++)
        {
            if (checked[solution[i * N + c] - 1] == true)
                isValid = false;
            checked[solution[i * N + c] - 1] = true;
        }
    }

    for (int s = 0; s < N; s++)
    {
        clearChecked(checked);
        int start = ((s / 3) * 3) * N + ((s % 3) * 3);
        for (int i = 0; i < N; i++)
        {
            int r = i / 3;
            int c = i % 3;
            if (checked[solution[start + r * N + c] - 1] == true)
                isValid = false;
            checked[solution[start + r * N + c] - 1] = true;
        }
    }

    return isValid;
}

__global__ void cudaDoBFS(Uint* oldBoard, Uint* newBoard, int boardsCount, int* lastBoard, int* empties, int isFinal)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < boardsCount)
    {
        // Indeks rozpatrywanego sudoku w tablicy
        int boardBegin = idx * BSIZE;
        int i = boardBegin;

        // Znalezienie pustego pola w sudoku
        while (i < boardBegin + N * N)
        {
            if (oldBoard[i] == 0)
                break;
            i++;
        }
        if (i == boardBegin + N * N)
            return;

        // Indeksy wiersza, kolumny i kwadratu w którym znajduje się puste pole
        int r = boardBegin + GET_ROW((i - boardBegin));
        int c = boardBegin + GET_COLUMN((i - boardBegin));
        int s = boardBegin + GET_SUBBOARD((i - boardBegin));
        for (int number = 1; number <= N; number++)
        {
            // Sprawdzenie, czy number można wpisać w wiersz, kolumnę i kwadrat
            if (CHECK_BIT(oldBoard[r], (number - 1)) && CHECK_BIT(oldBoard[c], (number - 1)) && CHECK_BIT(oldBoard[s], (number - 1)))
            {
                // Pobranie indeksu pierwszego wolnego miejsca na sudoku w tablicy newBoard
                int curBoard = atomicAdd(lastBoard, 1);
                int newBegin = curBoard * BSIZE;
                int emptiesIdx = curBoard * N * N;
                // Kopiowanie sudoku do newBoard
                for (int j = 0; j < BSIZE; j++)
                {
                    newBoard[newBegin + j] = oldBoard[boardBegin + j];
                    // Jeśli jesteśmy w ostatniej iteracji BFS wpisujemy wszystkie pozostałe puste pola do tablicy empties
                    if (isFinal == 1 && j < N * N && oldBoard[boardBegin + j] == 0 && boardBegin + j != i)
                    {
                        empties[emptiesIdx] = j;
                        emptiesIdx++;
                    }
                }
                // Uzupełnienie rozpatrywanego pola w skopiowanym sudoku
                newBoard[newBegin + i - boardBegin] = number;

                // Poprawienie masek bitowych dla wiersza, kolumny i kwadratu w skopiowanym sudoku
                CLEAR_BIT(newBoard[newBegin + r - boardBegin], (number - 1));
                CLEAR_BIT(newBoard[newBegin + c - boardBegin], (number - 1));
                CLEAR_BIT(newBoard[newBegin + s - boardBegin], (number - 1));
            }
        }
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void cudaDoDFS(Uint* board, int boardsCount, int* empties, int* outStatus, Uint* solution)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while ((*outStatus) == 0 && idx < boardsCount)
    {
        // Indeks rozpatrywanego sudoku w tablicy
        int boardBegin = idx * BSIZE;
        int emptiesBegin = idx * N * N;
        int emptiesIdx = emptiesBegin;

        // Dopoki istnieje puste pole w sudoku
        while (empties[emptiesIdx] >= 0)
        {
            // Indeks pierwszego pustego pola na podstawie tablicy empties
            int boardIdx = boardBegin + empties[emptiesIdx];

            // Indeksy wiersza, kolumny i kwadratu w którym znajduje się puste pole
            int r = boardBegin + GET_ROW((empties[emptiesIdx]));
            int c = boardBegin + GET_COLUMN((empties[emptiesIdx]));
            int s = boardBegin + GET_SUBBOARD((empties[emptiesIdx]));

            board[boardIdx]++;
            Uint number = board[boardIdx];
            // Sprawdz czy dla rozpatrywanego numeru sudoku będzie poprawne
            if (CHECK_BIT(board[r], (number - 1)) && CHECK_BIT(board[c], (number - 1)) && CHECK_BIT(board[s], (number - 1)))
            {
                // Popraw maski bitowe wiersza, kolumny i kwadratu dla wpisanego numeru
                CLEAR_BIT(board[r], (number - 1));
                CLEAR_BIT(board[c], (number - 1));
                CLEAR_BIT(board[s], (number - 1));

                // Idź do kolejnego pustego pola
                emptiesIdx++;
            }
            else
            {
                // Jeśli sprawdziliśmy już wszystkie możliwe liczby dla pola
                if (board[boardIdx] >= 9)
                {
                    // Zerujemy obecne pole i cofamy się do poprzedniego pustego pola
                    board[boardIdx] = 0;
                    emptiesIdx--;
                    // Jeśli to pierwsze puste pole, rozpatrywane sudoku nie ma rozwiązania
                    if (emptiesIdx < emptiesBegin)
                        break;

                    // Dla pola, do którego się cofamy, modyfikujemy maski bitowe wiersza, kolumny i kwadratu,
                    // tak aby odzwierciadlały stan sudoku gdy to pole jest puste
                    boardIdx = boardBegin + empties[emptiesIdx];
                    r = boardBegin + GET_ROW((empties[emptiesIdx]));
                    c = boardBegin + GET_COLUMN((empties[emptiesIdx]));
                    s = boardBegin + GET_SUBBOARD((empties[emptiesIdx]));
                    number = board[boardIdx];

                    SET_BIT(board[r], (number - 1));
                    SET_BIT(board[c], (number - 1));
                    SET_BIT(board[s], (number - 1));
                }
            }
        }

        // Jeśli przeszliśmy przez wszystkie puste pola przepisujemy 
        if (empties[emptiesIdx] < 0 && emptiesIdx > emptiesBegin)
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