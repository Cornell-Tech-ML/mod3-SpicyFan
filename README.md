# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py



(.venv) fylgzfls105@applwdeMBP-2 mod3-SpicyFan % python project/parallel_check.py
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (163)
  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                  | 
        out: Storage,                                                                          | 
        out_shape: Shape,                                                                      | 
        out_strides: Strides,                                                                  | 
        in_storage: Storage,                                                                   | 
        in_shape: Shape,                                                                       | 
        in_strides: Strides,                                                                   | 
    ) -> None:                                                                                 | 
        # Task 3.1 implementation.                                                             | 
        if np.array_equal(out_shape, in_shape) and np.array_equal(out_strides, in_strides):    | 
            for idx in prange(len(out)):-------------------------------------------------------| #2
                out[idx] = fn(in_storage[idx])                                                 | 
            return                                                                             | 
                                                                                               | 
        # Compute the total number of elements in the tensor                                   | 
        total_elements = len(out)                                                              | 
                                                                                               | 
         # Utilize prange for parallel execution in the main loop                              | 
        for idx in prange(total_elements):-----------------------------------------------------| #3
            # Convert flat index to multi-dimensional tensor index                             | 
            out_idx = np.zeros(MAX_DIMS, dtype=np.int32)---------------------------------------| #0
            to_index(idx, out_shape, out_idx)                                                  | 
                                                                                               | 
            # Adjust output index to match input index for broadcasting                        | 
            in_idx = np.zeros(MAX_DIMS, dtype=np.int32)----------------------------------------| #1
            broadcast_index(out_idx, out_shape, in_shape, in_idx)                              | 
                                                                                               | 
            # Get the positions in the respective storages                                     | 
            out_position = index_to_position(out_idx, out_strides)                             | 
            in_position = index_to_position(in_idx, in_strides)                                | 
                                                                                               | 
            # Execute the function and store the result                                        | 
            out[out_position] = fn(in_storage[in_position])                                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #3) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (187) 
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: in_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (183) 
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (223)
  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (223) 
------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                       | 
        out: Storage,                                                               | 
        out_shape: Shape,                                                           | 
        out_strides: Strides,                                                       | 
        a_storage: Storage,                                                         | 
        a_shape: Shape,                                                             | 
        a_strides: Strides,                                                         | 
        b_storage: Storage,                                                         | 
        b_shape: Shape,                                                             | 
        b_strides: Strides,                                                         | 
    ) -> None:                                                                      | 
        if (                                                                        | 
            np.array_equal(out_shape, a_shape)                                      | 
            and np.array_equal(out_shape, b_shape)                                  | 
            and np.array_equal(out_strides, a_strides)                              | 
            and np.array_equal(out_strides, b_strides)                              | 
        ):                                                                          | 
            for idx in prange(len(out)):--------------------------------------------| #7
                out[idx] = fn(a_storage[idx], b_storage[idx])                       | 
            return                                                                  | 
                                                                                    | 
        # Determine the total number of elements in the output tensor               | 
        total_elements = len(out)                                                   | 
                                                                                    | 
        # Execute the main loop in parallel using prange                            | 
        for idx in prange(total_elements):------------------------------------------| #8
            # Convert the flat index to a multi-dimensional index                   | 
            out_idx = np.zeros(MAX_DIMS, dtype=np.int32)----------------------------| #4
            to_index(idx, out_shape, out_idx)                                       | 
                                                                                    | 
            # Map output index to input indices for broadcasting                    | 
            a_idx = np.zeros(MAX_DIMS, dtype=np.int32)------------------------------| #5
            b_idx = np.zeros(MAX_DIMS, dtype=np.int32)------------------------------| #6
            broadcast_index(out_idx, out_shape, a_shape, a_idx)                     | 
            broadcast_index(out_idx, out_shape, b_shape, b_idx)                     | 
                                                                                    | 
            # Get positions in the respective storages                              | 
            out_position = index_to_position(out_idx, out_strides)                  | 
            a_position = index_to_position(a_idx, a_strides)                        | 
            b_position = index_to_position(b_idx, b_strides)                        | 
                                                                                    | 
            # Apply the binary function and save the result                         | 
            out[out_position] = fn(a_storage[a_position], b_storage[b_position])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--5 has the following loops fused into it:
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial)
   +--5 (serial, fused with loop(s): 6)


 
Parallel region 0 (loop #8) had 1 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (250) 
is hoisted out of the parallel loop labelled #8 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (254) 
is hoisted out of the parallel loop labelled #8 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: a_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (255) 
is hoisted out of the parallel loop labelled #8 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: b_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (291)
  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (291) 
------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                    | 
        out: Storage,                                                               | 
        out_shape: Shape,                                                           | 
        out_strides: Strides,                                                       | 
        a_storage: Storage,                                                         | 
        a_shape: Shape,                                                             | 
        a_strides: Strides,                                                         | 
        reduce_dim: int,                                                            | 
    ) -> None:                                                                      | 
        # Determine the total number of elements in the output tensor               | 
        total_elements = len(out)                                                   | 
        # Get the size of the dimension to be reduced                               | 
        reduce_size = a_shape[reduce_dim]                                           | 
                                                                                    | 
        # Use prange to parallelize the main loop                                   | 
        for idx in prange(total_elements):------------------------------------------| #11
            # Convert the flat index to a multi-dimensional tensor index            | 
            out_idx = np.zeros(MAX_DIMS, dtype=np.int32)----------------------------| #9
            to_index(idx, out_shape, out_idx)                                       | 
                                                                                    | 
            # Create an index to access elements from the input tensor              | 
            a_idx = np.zeros(len(a_shape), dtype=np.int32)--------------------------| #10
            for dim in range(len(out_shape)):                                       | 
                a_idx[dim] = out_idx[dim]                                           | 
                                                                                    | 
            # Calculate the position in the output storage                          | 
            out_position = index_to_position(out_idx, out_strides)                  | 
                                                                                    | 
            # Perform the reduction along the specified dimension                   | 
            for j in range(reduce_size):                                            | 
                a_idx[reduce_dim] = j                                               | 
                a_position = index_to_position(a_idx, a_strides)                    | 
                out[out_position] = fn(out[out_position], a_storage[a_position])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #11, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--11 is a parallel loop
   +--9 --> rewritten as a serial loop
   +--10 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--9 (parallel)
   +--10 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--9 (serial)
   +--10 (serial)


 
Parallel region 0 (loop #11) had 0 loop(s) fused and 2 loop(s) serialized as 
part of the larger parallel loop (#11).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (308) 
is hoisted out of the parallel loop labelled #11 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (312) 
is hoisted out of the parallel loop labelled #11 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: a_idx = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (328)
  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (328) 
--------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                              | 
    out: Storage,                                                         | 
    out_shape: Shape,                                                     | 
    out_strides: Strides,                                                 | 
    a_storage: Storage,                                                   | 
    a_shape: Shape,                                                       | 
    a_strides: Strides,                                                   | 
    b_storage: Storage,                                                   | 
    b_shape: Shape,                                                       | 
    b_strides: Strides,                                                   | 
) -> None:                                                                | 
    """NUMBA tensor matrix multiply function.                             | 
                                                                          | 
    Should work for any tensor shapes that broadcast as long as           | 
                                                                          | 
    ```                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                     | 
    ```                                                                   | 
                                                                          | 
    Optimizations:                                                        | 
                                                                          | 
    * Outer loop in parallel                                              | 
    * No index buffers or function calls                                  | 
    * Inner loop should have no global writes, 1 multiply.                | 
                                                                          | 
                                                                          | 
    Args:                                                                 | 
    ----                                                                  | 
        out (Storage): storage for `out` tensor                           | 
        out_shape (Shape): shape for `out` tensor                         | 
        out_strides (Strides): strides for `out` tensor                   | 
        a_storage (Storage): storage for `a` tensor                       | 
        a_shape (Shape): shape for `a` tensor                             | 
        a_strides (Strides): strides for `a` tensor                       | 
        b_storage (Storage): storage for `b` tensor                       | 
        b_shape (Shape): shape for `b` tensor                             | 
        b_strides (Strides): strides for `b` tensor                       | 
                                                                          | 
    Returns:                                                              | 
    -------                                                               | 
        None : Fills in `out`                                             | 
                                                                          | 
    """                                                                   | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                | 
                                                                          | 
    # Extract the dimensions                                              | 
    batch_size = out_shape[0]                                             | 
    rows = out_shape[1]                                                   | 
    cols = out_shape[2]                                                   | 
    reduce_dim = a_shape[2]                                               | 
                                                                          | 
    # Parallelize over the outer loops                                    | 
    for batch in prange(batch_size):--------------------------------------| #13
        for row in prange(rows):------------------------------------------| #12
            for col in range(cols):                                       | 
                # Compute the output position                             | 
                out_pos = (                                               | 
                    batch * out_strides[0] +                              | 
                    row * out_strides[1] +                                | 
                    col * out_strides[2]                                  | 
                )                                                         | 
                                                                          | 
                # Initialize the accumulator                              | 
                accumulator = 0.0                                         | 
                                                                          | 
                # Inner loop for reduction                                | 
                for k in range(reduce_dim):                               | 
                    # Compute positions in a and b storages               | 
                    a_pos = (                                             | 
                        batch * a_batch_stride +                          | 
                        row * a_strides[1] +                              | 
                        k * a_strides[2]                                  | 
                    )                                                     | 
                    b_pos = (                                             | 
                        batch * b_batch_stride +                          | 
                        k * b_strides[1] +                                | 
                        col * b_strides[2]                                | 
                    )                                                     | 
                    # Multiply and accumulate                             | 
                    accumulator += a_storage[a_pos] * b_storage[b_pos]    | 
                                                                          | 
                # Write the final result                                  | 
                out[out_pos] = accumulator                                | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)


 
Parallel region 0 (loop #13) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
(.venv) fylgzfls105@applwdeMBP-2 mod3-SpicyFan % pytest -m task3_1               
========================================================== test session starts ===========================================================
platform darwin -- Python 3.12.5, pytest-8.3.2, pluggy-1.5.0
rootdir: /Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan
configfile: pyproject.toml
plugins: hypothesis-6.54.0, env-1.1.4
collected 53 items / 2 deselected / 51 selected                                                                                          

tests/test_tensor_general.py ...................................................                                                   [100%]

=================================================== 51 passed, 2 deselected in 51.42s ====================================================
(.venv) fylgzfls105@applwdeMBP-2 mod3-SpicyFan % pytest -m task3_2
========================================================== test session starts ===========================================================
platform darwin -- Python 3.12.5, pytest-8.3.2, pluggy-1.5.0
rootdir: /Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan
configfile: pyproject.toml
plugins: hypothesis-6.54.0, env-1.1.4
collected 53 items / 51 deselected / 2 selected                                                                                          

tests/test_tensor_general.py ..                                                                                                    [100%]

=================================================== 2 passed, 51 deselected in 14.63s ====================================================
(.venv) fylgzfls105@applwdeMBP-2 mod3-SpicyFan % python project/parallel_check.py
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (163)
  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                  | 
        out: Storage,                                                                          | 
        out_shape: Shape,                                                                      | 
        out_strides: Strides,                                                                  | 
        in_storage: Storage,                                                                   | 
        in_shape: Shape,                                                                       | 
        in_strides: Strides,                                                                   | 
    ) -> None:                                                                                 | 
        # Task 3.1 implementation.                                                             | 
        if np.array_equal(out_shape, in_shape) and np.array_equal(out_strides, in_strides):    | 
            for idx in prange(len(out)):-------------------------------------------------------| #2
                out[idx] = fn(in_storage[idx])                                                 | 
            return                                                                             | 
                                                                                               | 
        # Compute the total number of elements in the tensor                                   | 
        total_elements = len(out)                                                              | 
                                                                                               | 
         # Utilize prange for parallel execution in the main loop                              | 
        for idx in prange(total_elements):-----------------------------------------------------| #3
            # Convert flat index to multi-dimensional tensor index                             | 
            out_idx = np.zeros(MAX_DIMS, dtype=np.int32)---------------------------------------| #0
            to_index(idx, out_shape, out_idx)                                                  | 
                                                                                               | 
            # Adjust output index to match input index for broadcasting                        | 
            in_idx = np.zeros(MAX_DIMS, dtype=np.int32)----------------------------------------| #1
            broadcast_index(out_idx, out_shape, in_shape, in_idx)                              | 
                                                                                               | 
            # Get the positions in the respective storages                                     | 
            out_position = index_to_position(out_idx, out_strides)                             | 
            in_position = index_to_position(in_idx, in_strides)                                | 
                                                                                               | 
            # Execute the function and store the result                                        | 
            out[out_position] = fn(in_storage[in_position])                                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #3) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (187) 
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: in_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (183) 
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (223)
  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (223) 
------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                       | 
        out: Storage,                                                               | 
        out_shape: Shape,                                                           | 
        out_strides: Strides,                                                       | 
        a_storage: Storage,                                                         | 
        a_shape: Shape,                                                             | 
        a_strides: Strides,                                                         | 
        b_storage: Storage,                                                         | 
        b_shape: Shape,                                                             | 
        b_strides: Strides,                                                         | 
    ) -> None:                                                                      | 
        if (                                                                        | 
            np.array_equal(out_shape, a_shape)                                      | 
            and np.array_equal(out_shape, b_shape)                                  | 
            and np.array_equal(out_strides, a_strides)                              | 
            and np.array_equal(out_strides, b_strides)                              | 
        ):                                                                          | 
            for idx in prange(len(out)):--------------------------------------------| #7
                out[idx] = fn(a_storage[idx], b_storage[idx])                       | 
            return                                                                  | 
                                                                                    | 
        # Determine the total number of elements in the output tensor               | 
        total_elements = len(out)                                                   | 
                                                                                    | 
        # Execute the main loop in parallel using prange                            | 
        for idx in prange(total_elements):------------------------------------------| #8
            # Convert the flat index to a multi-dimensional index                   | 
            out_idx = np.zeros(MAX_DIMS, dtype=np.int32)----------------------------| #4
            to_index(idx, out_shape, out_idx)                                       | 
                                                                                    | 
            # Map output index to input indices for broadcasting                    | 
            a_idx = np.zeros(MAX_DIMS, dtype=np.int32)------------------------------| #5
            b_idx = np.zeros(MAX_DIMS, dtype=np.int32)------------------------------| #6
            broadcast_index(out_idx, out_shape, a_shape, a_idx)                     | 
            broadcast_index(out_idx, out_shape, b_shape, b_idx)                     | 
                                                                                    | 
            # Get positions in the respective storages                              | 
            out_position = index_to_position(out_idx, out_strides)                  | 
            a_position = index_to_position(a_idx, a_strides)                        | 
            b_position = index_to_position(b_idx, b_strides)                        | 
                                                                                    | 
            # Apply the binary function and save the result                         | 
            out[out_position] = fn(a_storage[a_position], b_storage[b_position])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--5 has the following loops fused into it:
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial)
   +--5 (serial, fused with loop(s): 6)


 
Parallel region 0 (loop #8) had 1 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (250) 
is hoisted out of the parallel loop labelled #8 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (254) 
is hoisted out of the parallel loop labelled #8 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: a_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (255) 
is hoisted out of the parallel loop labelled #8 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: b_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (291)
  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (291) 
------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                    | 
        out: Storage,                                                               | 
        out_shape: Shape,                                                           | 
        out_strides: Strides,                                                       | 
        a_storage: Storage,                                                         | 
        a_shape: Shape,                                                             | 
        a_strides: Strides,                                                         | 
        reduce_dim: int,                                                            | 
    ) -> None:                                                                      | 
        # Determine the total number of elements in the output tensor               | 
        total_elements = len(out)                                                   | 
        # Get the size of the dimension to be reduced                               | 
        reduce_size = a_shape[reduce_dim]                                           | 
                                                                                    | 
        # Use prange to parallelize the main loop                                   | 
        for idx in prange(total_elements):------------------------------------------| #11
            # Convert the flat index to a multi-dimensional tensor index            | 
            out_idx = np.zeros(MAX_DIMS, dtype=np.int32)----------------------------| #9
            to_index(idx, out_shape, out_idx)                                       | 
                                                                                    | 
            # Create an index to access elements from the input tensor              | 
            a_idx = np.zeros(len(a_shape), dtype=np.int32)--------------------------| #10
            for dim in range(len(out_shape)):                                       | 
                a_idx[dim] = out_idx[dim]                                           | 
                                                                                    | 
            # Calculate the position in the output storage                          | 
            out_position = index_to_position(out_idx, out_strides)                  | 
                                                                                    | 
            # Perform the reduction along the specified dimension                   | 
            for j in range(reduce_size):                                            | 
                a_idx[reduce_dim] = j                                               | 
                a_position = index_to_position(a_idx, a_strides)                    | 
                out[out_position] = fn(out[out_position], a_storage[a_position])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #11, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--11 is a parallel loop
   +--9 --> rewritten as a serial loop
   +--10 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--9 (parallel)
   +--10 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--9 (serial)
   +--10 (serial)


 
Parallel region 0 (loop #11) had 0 loop(s) fused and 2 loop(s) serialized as 
part of the larger parallel loop (#11).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (308) 
is hoisted out of the parallel loop labelled #11 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (312) 
is hoisted out of the parallel loop labelled #11 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: a_idx = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (328)
  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/fylgzfls105/Desktop/CS5781 MLE/mod3-SpicyFan/minitorch/fast_ops.py (328) 
--------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                              | 
    out: Storage,                                                         | 
    out_shape: Shape,                                                     | 
    out_strides: Strides,                                                 | 
    a_storage: Storage,                                                   | 
    a_shape: Shape,                                                       | 
    a_strides: Strides,                                                   | 
    b_storage: Storage,                                                   | 
    b_shape: Shape,                                                       | 
    b_strides: Strides,                                                   | 
) -> None:                                                                | 
    """NUMBA tensor matrix multiply function.                             | 
                                                                          | 
    Should work for any tensor shapes that broadcast as long as           | 
                                                                          | 
    ```                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                     | 
    ```                                                                   | 
                                                                          | 
    Optimizations:                                                        | 
                                                                          | 
    * Outer loop in parallel                                              | 
    * No index buffers or function calls                                  | 
    * Inner loop should have no global writes, 1 multiply.                | 
                                                                          | 
                                                                          | 
    Args:                                                                 | 
    ----                                                                  | 
        out (Storage): storage for `out` tensor                           | 
        out_shape (Shape): shape for `out` tensor                         | 
        out_strides (Strides): strides for `out` tensor                   | 
        a_storage (Storage): storage for `a` tensor                       | 
        a_shape (Shape): shape for `a` tensor                             | 
        a_strides (Strides): strides for `a` tensor                       | 
        b_storage (Storage): storage for `b` tensor                       | 
        b_shape (Shape): shape for `b` tensor                             | 
        b_strides (Strides): strides for `b` tensor                       | 
                                                                          | 
    Returns:                                                              | 
    -------                                                               | 
        None : Fills in `out`                                             | 
                                                                          | 
    """                                                                   | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                | 
                                                                          | 
    # Extract the dimensions                                              | 
    batch_size = out_shape[0]                                             | 
    rows = out_shape[1]                                                   | 
    cols = out_shape[2]                                                   | 
    reduce_dim = a_shape[2]                                               | 
                                                                          | 
    # Parallelize over the outer loops                                    | 
    for batch in prange(batch_size):--------------------------------------| #13
        for row in prange(rows):------------------------------------------| #12
            for col in range(cols):                                       | 
                # Compute the output position                             | 
                out_pos = (                                               | 
                    batch * out_strides[0] +                              | 
                    row * out_strides[1] +                                | 
                    col * out_strides[2]                                  | 
                )                                                         | 
                                                                          | 
                # Initialize the accumulator                              | 
                accumulator = 0.0                                         | 
                                                                          | 
                # Inner loop for reduction                                | 
                for k in range(reduce_dim):                               | 
                    # Compute positions in a and b storages               | 
                    a_pos = (                                             | 
                        batch * a_batch_stride +                          | 
                        row * a_strides[1] +                              | 
                        k * a_strides[2]                                  | 
                    )                                                     | 
                    b_pos = (                                             | 
                        batch * b_batch_stride +                          | 
                        k * b_strides[1] +                                | 
                        col * b_strides[2]                                | 
                    )                                                     | 
                    # Multiply and accumulate                             | 
                    accumulator += a_storage[a_pos] * b_storage[b_pos]    | 
                                                                          | 
                # Write the final result                                  | 
                out[out_pos] = accumulator                                | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)


 
Parallel region 0 (loop #13) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None