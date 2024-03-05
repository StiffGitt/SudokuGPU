# SudokuGPU
A CUDA parallel approach for fast solving of Sudoku puzzles.

The algorithm consists of combination of breadth first and depth first search algorithms computed sequentially, this concept allows to first compute in parallel thousands of partly solved boards, then launch 
parallel DFS which finds the answer amongst the boards. Both the BFS and DFS are implemented inside a CUDA kernel.
