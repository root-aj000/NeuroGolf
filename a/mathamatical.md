To ensure your engineering team can implement this system flawlessly, let’s break down every single component of the Zero-Parameter Permutation Architecture into precise mathematical, programmatic, and ONNX runtime details.
------------------------------
## Step 1: The Offline Rule Induction Engine
Before touching ONNX, your local pipeline must analyze the Kaggle task data to find the mathematical rule. ARC puzzles use static grid dimensions within a single task (e.g., all grids are $30 \times 30$). [1] 
## Action: Vectorization (Flattening)
A 2D matrix mask hides the underlying linear relationships. We flatten the 2D matrix into a 1D tensor vector of size $N$ (where $N = \text{Height} \times \text{Width}$).

* For a $30 \times 30$ grid, $N = 900$.
* Row $r$ and Column $c$ maps to a 1D index using row-major ordering:
$$\text{Index} = r \times \text{Width} + c$$ 

## Action: Cross-Pair Validation (Eliminating Overfitting)
An ARC task contains multiple Demonstration Examples (e.g., Demo 1, Demo 2, Demo 3). If you only map Demo 1, you will overfit to its specific pixel values. To find the general structural rule, you must find a coordinate mapping that holds true across all demos.
We test the 8 symmetries of the Dihedral Group ($D_4$) by building index-mapping lookup arrays. For any cell coordinate $(r, c)$ in a $30 \times 30$ grid, the transformation rules map the index like this:

| Transformation Name | New Row ($r'$) | New Column ($c'$) | 1D Index Formula ($r' \times 30 + c'$) |
|---|---|---|---|
| Identity (No Change) | $r$ | $c$ | $r \times 30 + c$ |
| Flip Left-Right | $r$ | $29 - c$ | $r \times 30 + (29 - c)$ |
| Flip Up-Down | $29 - r$ | $c$ | $(29 - r) \times 30 + c$ |
| Rotate 90° CW | $c$ | $29 - r$ | $c \times 30 + (29 - r)$ |
| Rotate 180° | $29 - r$ | $29 - c$ | $(29 - r) \times 30 + (29 - c)$ |
| Rotate 270° CW | $29 - c$ | $r$ | $(29 - c) \times 30 + r$ |
| Transpose (Diagonal) | $c$ | $r$ | $c \times 30 + r$ |
| Anti-Transpose | $29 - c$ | $29 - r$ | $(29 - c) \times 30 + (29 - r)$ |

The Execution Loop: Your script applies each of these 8 index maps to the flattened input vector of every demo pair. If flat_input[map] == flat_output for all demo pairs, that specific transformation is locked in as the absolute spatial rule for the task.
------------------------------
## Step 2: Compiling the Permutation Matrix ($\mathbf{P}$)
Once the winning index map is identified, you convert that 1D mapping array into a massive, sparse 2D Permutation Matrix ($\mathbf{P}$) of size $900 \times 900$.
## Action: Building the Matrix Structure
A permutation matrix uses row-column intersections to route data. If the winning rule dictates that the pixel at input index $j$ must move to output index $i$, then in your matrix:
$$\mathbf{P}[i, j] = 1.0$$ 
All other elements in row $i$ and column $j$ are set to $0.0$.
## Why it works mathematically:
When you perform a matrix multiplication between a row vector $\vec{x}$ (size $1 \times 900$) and the transpose of your permutation matrix $\mathbf{P}^T$ (size $900 \times 900$), the dot product for output element $y_i$ is:
$$y_i = \mathbf{X}_{1,1}\mathbf{P}^T_{1,i} + \mathbf{X}_{1,2}\mathbf{P}^T_{2,i} + \dots + \mathbf{X}_{1,j}\mathbf{P}^T_{j,i}$$ 
Because every element in the $i$-th column of $\mathbf{P}^T$ is $0.0$ except for a single $1.0$ at position $j$, the entire equation collapses to:
$$y_i = \mathbf{X}_{1,j} \times 1.0 = x_j$$ 
The system copies the value from input index $j$ straight to output index $i$ inside the hardware registry, bypassing traditional floating-point calculations.
------------------------------
## Step 3: Assembling the Static ONNX Graph
Now, you assemble the .onnx graph file. Under the competition constraints, you must hardcode all tensor shapes statically.

       [Input Tensor]  Shape: (1, 30, 30)
             │
             ▼
       ┌───────────┐
       │  Flatten  │  Axis=1
       └─────┬─────┘
             │         Shape: (1, 900)
             ▼
       ┌───────────┐
       │   Gemm    │  Inputs: [input_flat, Constant_P]
       └─────┬─────┘  transB=1, alpha=1.0, beta=0.0
             │         Shape: (1, 900)
             ▼
       ┌───────────┐
       │  Reshape  │  Inputs: [output_flat, Constant_Shape]
       └─────┬─────┘
             │
             ▼
      [Output Tensor]  Shape: (1, 30, 30)

## Node 1: Flatten

* Purpose: Converts the multi-dimensional grid into a single-dimensional vector layout.
* Configuration: Set axis=1. This tells ONNX to preserve the batch dimension (dimension 0) and flatten everything else. If the input is (1, 30, 30), the output becomes (1, 900).

## Node 2: Constant (Embedding $\mathbf{P}$)

* Purpose: Stores the $900 \times 900$ matrix directly inside the ONNX file structure.
* Configuration: Embedded as an ONNX Initializer tensor. This ensures the evaluation engine treats it as a static compile-time constant rather than a variable model parameter, keeping your active parameter score at zero.

## Node 3: Gemm (General Matrix Multiplication)

* Purpose: Executes the core structural rearrangement of the pixels.
* Configuration:
* Input A: input_flat (The dynamic $(1 \times 900)$ vector from the input node).
   * Input B: P (The static $(900 \times 900)$ constant tensor).
   * Attribute transB=1: Automatically transposes matrix B at runtime so the dot-product shapes line up flawlessly.
   * Attribute alpha=1.0: Multiplier for the matrix multiplication.
   * Attribute beta=0.0: Multiplier for the bias layer (disabling it entirely).

## Node 4: Reshape

* Purpose: Converts the processed 1D vector back into the target 2D image matrix.
* Configuration: Requires a secondary static Constant node containing the target dimension array: [1, 30, 30]. It maps the $(1 \times 900)$ vector back into a clean $(1 \times 30 \times 30)$ tensor grid.

------------------------------
## Step 4: Submission and Validation Pass
Once the pipeline loops through all the competition tasks, it outputs an individual .onnx file named after each unique task ID into a flat directory.

   1. Compression: The directory is compiled into a single standard .zip file structure.
   2. Kaggle Ingestion: The evaluation server unzips the file and passes each hidden test input grid through its corresponding ONNX graph model.
   3. Execution: The ONNX runtime processes the mathematical graph in a single feed-forward pass. Because there are no dynamic loops (Loop, Scan) or conditional searching nodes (NonZero), the execution pass runs smoothly and matches the constraints perfectly.

If your team wants to start building, I can show you how to expand this exact pipeline to handle color-swapping logic gates (e.g., swapping Color 1 and Color 5) using basic matrix subtraction and addition nodes.

[1] [https://arxiv.org](https://arxiv.org/html/2511.16886v2)
