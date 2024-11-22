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

TASK 3.5 OUTPUT RESULT
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05

Time per epoch: 20.3643s Epoch  0  loss  11.06773244758223 correct 21
Time per epoch: 0.1679s Epoch  10  loss  5.659140387797978 correct 41
Time per epoch: 0.1685s Epoch  20  loss  3.8413372097247303 correct 46
Time per epoch: 0.1667s Epoch  30  loss  3.8800617054282305 correct 45
Time per epoch: 0.1626s Epoch  40  loss  2.582912116117575 correct 48
Time per epoch: 0.1470s Epoch  50  loss  2.9172788300567305 correct 47
Time per epoch: 0.1432s Epoch  60  loss  2.0459315839088434 correct 49
Time per epoch: 0.1439s Epoch  70  loss  2.3061732592124584 correct 49
Time per epoch: 0.1491s Epoch  80  loss  2.3626649950130116 correct 49
Time per epoch: 0.1459s Epoch  90  loss  1.497758091976635 correct 49
Time per epoch: 0.1481s Epoch  100  loss  1.6854292606732486 correct 49
Time per epoch: 0.1445s Epoch  110  loss  1.9196163069630463 correct 49
Time per epoch: 0.1430s Epoch  120  loss  1.2382683352224573 correct 48
Time per epoch: 0.1417s Epoch  130  loss  1.0582189485698614 correct 49
Time per epoch: 0.1468s Epoch  140  loss  1.3900658645646407 correct 49
Time per epoch: 0.1455s Epoch  150  loss  0.9135651040439952 correct 49
Time per epoch: 0.1529s Epoch  160  loss  0.4441583046402097 correct 50
Time per epoch: 0.1497s Epoch  170  loss  0.5438905486249339 correct 50
Time per epoch: 0.1551s Epoch  180  loss  1.5068949550210424 correct 50
Time per epoch: 0.1422s Epoch  190  loss  1.0584845324385217 correct 50
Time per epoch: 0.1434s Epoch  200  loss  0.8212609147281357 correct 50
Time per epoch: 0.1478s Epoch  210  loss  1.0820589871574653 correct 50
Time per epoch: 0.1477s Epoch  220  loss  0.31251011399676054 correct 49
Time per epoch: 0.1735s Epoch  230  loss  0.8145811458936857 correct 49
Time per epoch: 0.1652s Epoch  240  loss  1.068018305433477 correct 48
Time per epoch: 0.1503s Epoch  250  loss  0.9868946000646879 correct 49
Time per epoch: 0.1502s Epoch  260  loss  1.1580080304495597 correct 49
Time per epoch: 0.1431s Epoch  270  loss  1.231968152895885 correct 49
Time per epoch: 0.1471s Epoch  280  loss  0.7282667758174959 correct 50
Time per epoch: 0.1429s Epoch  290  loss  0.02902022571835588 correct 49
Time per epoch: 0.1447s Epoch  300  loss  0.26598368832182506 correct 50
Time per epoch: 0.1463s Epoch  310  loss  0.19461522279672405 correct 49
Time per epoch: 0.1506s Epoch  320  loss  0.9263513848657716 correct 50
Time per epoch: 0.1430s Epoch  330  loss  0.19467705346170405 correct 50
Time per epoch: 0.1398s Epoch  340  loss  1.7492564906011001 correct 48
Time per epoch: 0.1456s Epoch  350  loss  0.7310533892757316 correct 50
Time per epoch: 0.1621s Epoch  360  loss  0.0588253499872779 correct 50
Time per epoch: 0.2899s Epoch  370  loss  1.0920026290652176 correct 49
Time per epoch: 0.1441s Epoch  380  loss  1.4030888965627903 correct 49
Time per epoch: 0.1556s Epoch  390  loss  0.23108849123774902 correct 50
Time per epoch: 0.1568s Epoch  400  loss  0.3211778372978309 correct 49
Time per epoch: 0.1486s Epoch  410  loss  0.5381079630910399 correct 49
Time per epoch: 0.1890s Epoch  420  loss  0.24425670440531852 correct 50
Time per epoch: 0.1611s Epoch  430  loss  0.11149430140707434 correct 50
Time per epoch: 0.1543s Epoch  440  loss  0.33009639560352955 correct 50
Time per epoch: 0.1532s Epoch  450  loss  0.7947291087058114 correct 49
Time per epoch: 0.1596s Epoch  460  loss  0.030441856292795014 correct 50
Time per epoch: 0.1596s Epoch  470  loss  0.19033265078104386 correct 50
Time per epoch: 0.1857s Epoch  480  loss  0.8056539587243576 correct 50
Time per epoch: 0.1584s Epoch  490  loss  0.9966411870276067 correct 49

python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05

Time per epoch: 18.6842s Epoch  0  loss  6.194328893188165 correct 35
Time per epoch: 0.2217s Epoch  10  loss  3.4735482633985386 correct 42
Time per epoch: 0.2265s Epoch  20  loss  5.633887876031119 correct 44
Time per epoch: 0.1915s Epoch  30  loss  1.6933134911380001 correct 46
Time per epoch: 0.1556s Epoch  40  loss  2.389687516212882 correct 45
Time per epoch: 0.1590s Epoch  50  loss  2.8056204937278397 correct 48
Time per epoch: 0.1626s Epoch  60  loss  0.8578507431660802 correct 46
Time per epoch: 0.1651s Epoch  70  loss  2.616533379280961 correct 48
Time per epoch: 0.1603s Epoch  80  loss  0.6559450658093081 correct 47
Time per epoch: 0.1699s Epoch  90  loss  1.559052031635239 correct 48
Time per epoch: 0.1402s Epoch  100  loss  1.3616576919816248 correct 49
Time per epoch: 0.1449s Epoch  110  loss  2.7573506376963164 correct 49
Time per epoch: 0.1413s Epoch  120  loss  0.15417451104080088 correct 48
Time per epoch: 0.1433s Epoch  130  loss  1.378118819532899 correct 50
Time per epoch: 0.1423s Epoch  140  loss  1.7350572036536085 correct 50
Time per epoch: 0.1393s Epoch  150  loss  2.032088532931385 correct 48
Time per epoch: 0.1426s Epoch  160  loss  0.15178743258427788 correct 50
Time per epoch: 0.1351s Epoch  170  loss  0.39744750834487624 correct 50
Time per epoch: 0.1408s Epoch  180  loss  0.26905514160150423 correct 50
Time per epoch: 0.1445s Epoch  190  loss  0.9370555709090931 correct 50
Time per epoch: 0.1676s Epoch  200  loss  0.5723827112637869 correct 50
Time per epoch: 0.1486s Epoch  210  loss  0.5759345019128032 correct 50
Time per epoch: 0.1424s Epoch  220  loss  0.8412226746621799 correct 49
Time per epoch: 0.1457s Epoch  230  loss  0.544589081404538 correct 50
Time per epoch: 0.1362s Epoch  240  loss  0.5851133812334776 correct 50
Time per epoch: 0.1434s Epoch  250  loss  0.20207069600577376 correct 50
Time per epoch: 0.1405s Epoch  260  loss  0.6037661396728916 correct 50
Time per epoch: 0.1382s Epoch  270  loss  0.3188774836840605 correct 50
Time per epoch: 0.1336s Epoch  280  loss  0.30240806266020126 correct 50
Time per epoch: 0.1386s Epoch  290  loss  0.05889846701410852 correct 50
Time per epoch: 0.1379s Epoch  300  loss  0.2836536788902657 correct 50
Time per epoch: 0.1402s Epoch  310  loss  0.16567923364822362 correct 50
Time per epoch: 0.1370s Epoch  320  loss  0.6497677746730013 correct 50
Time per epoch: 0.1385s Epoch  330  loss  0.09912266341273758 correct 50
Time per epoch: 0.1423s Epoch  340  loss  0.07201850307921105 correct 50
Time per epoch: 0.1349s Epoch  350  loss  0.6458865158261548 correct 50
Time per epoch: 0.1354s Epoch  360  loss  0.20634233666760202 correct 50
Time per epoch: 0.1326s Epoch  370  loss  0.28824169892024926 correct 50
Time per epoch: 0.1414s Epoch  380  loss  0.1701364674298265 correct 50
Time per epoch: 0.1450s Epoch  390  loss  0.12113417455916656 correct 50
Time per epoch: 0.1570s Epoch  400  loss  0.1542552000827999 correct 50
Time per epoch: 0.1727s Epoch  410  loss  0.1835636146901601 correct 50
Time per epoch: 0.1646s Epoch  420  loss  0.04404936809801579 correct 50
Time per epoch: 0.1411s Epoch  430  loss  0.27054869147621025 correct 50
Time per epoch: 0.1385s Epoch  440  loss  0.04814825591714145 correct 50
Time per epoch: 0.1416s Epoch  450  loss  0.16464982822358668 correct 50
Time per epoch: 0.1625s Epoch  460  loss  0.15922047319690086 correct 50
Time per epoch: 0.1838s Epoch  470  loss  0.2140735145951801 correct 50
Time per epoch: 0.1841s Epoch  480  loss  0.09827612730918989 correct 50
Time per epoch: 0.1624s Epoch  490  loss  0.1006252136084656 correct 50


python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05

Time per epoch: 17.9962s Epoch  0  loss  5.34911382207067 correct 40
Time per epoch: 0.1675s Epoch  10  loss  0.8926716276067259 correct 46
Time per epoch: 0.1448s Epoch  20  loss  0.4228893353221198 correct 47
Time per epoch: 0.1482s Epoch  30  loss  2.383311713161959 correct 48
Time per epoch: 0.1436s Epoch  40  loss  3.0067264094766863 correct 48
Time per epoch: 0.1374s Epoch  50  loss  1.0907225946038306 correct 48
Time per epoch: 0.1351s Epoch  60  loss  0.7584966264108911 correct 48
Time per epoch: 0.1385s Epoch  70  loss  1.0758906897301839 correct 49
Time per epoch: 0.1472s Epoch  80  loss  2.2193372777759386 correct 49
Time per epoch: 0.1457s Epoch  90  loss  1.9933114769443745 correct 48
Time per epoch: 0.1351s Epoch  100  loss  0.317703393250315 correct 50
Time per epoch: 0.1402s Epoch  110  loss  0.9732959427961927 correct 48
Time per epoch: 0.1429s Epoch  120  loss  3.3063304257863173 correct 48
Time per epoch: 0.1359s Epoch  130  loss  0.12611619244078387 correct 50
Time per epoch: 0.1419s Epoch  140  loss  0.28217299133086743 correct 50
Time per epoch: 0.1417s Epoch  150  loss  0.423395925075702 correct 48
Time per epoch: 0.1753s Epoch  160  loss  0.7575786111416728 correct 48
Time per epoch: 0.1676s Epoch  170  loss  1.5736697158924597 correct 50
Time per epoch: 0.1481s Epoch  180  loss  2.677104153580874 correct 48
Time per epoch: 0.1656s Epoch  190  loss  0.45164641287117 correct 48
Time per epoch: 0.1388s Epoch  200  loss  1.3697053261109609 correct 50
Time per epoch: 0.1448s Epoch  210  loss  1.5727754692207034 correct 50
Time per epoch: 0.1490s Epoch  220  loss  1.1630340063821727 correct 50
Time per epoch: 0.1370s Epoch  230  loss  0.0916255753975597 correct 48
Time per epoch: 0.1438s Epoch  240  loss  1.2154630342368344 correct 49
Time per epoch: 0.1486s Epoch  250  loss  1.426285110240004 correct 48
Time per epoch: 0.1495s Epoch  260  loss  1.1711665726496285 correct 49
Time per epoch: 0.1413s Epoch  270  loss  0.8065090542611023 correct 48
Time per epoch: 0.1394s Epoch  280  loss  1.0203651265776883 correct 48
Time per epoch: 0.1550s Epoch  290  loss  0.605092665505089 correct 48
Time per epoch: 0.1448s Epoch  300  loss  0.578248004182146 correct 49
Time per epoch: 0.1485s Epoch  310  loss  0.005051935191083199 correct 48
Time per epoch: 0.1418s Epoch  320  loss  1.0550356042058717 correct 49
Time per epoch: 0.1477s Epoch  330  loss  1.2732821346070256 correct 49
Time per epoch: 0.1472s Epoch  340  loss  0.1898547575863671 correct 49
Time per epoch: 0.1403s Epoch  350  loss  0.009793714628772895 correct 48
Time per epoch: 0.1640s Epoch  360  loss  0.05500877884737299 correct 50
Time per epoch: 0.1510s Epoch  370  loss  0.323365217856106 correct 48
Time per epoch: 0.1534s Epoch  380  loss  1.0776155028472423 correct 50
Time per epoch: 0.1387s Epoch  390  loss  1.1683994230898413 correct 49
Time per epoch: 0.1377s Epoch  400  loss  0.010826814364230644 correct 49
Time per epoch: 0.1433s Epoch  410  loss  0.03105735508760475 correct 50
Time per epoch: 0.1503s Epoch  420  loss  1.0877433705286759 correct 49
Time per epoch: 0.1474s Epoch  430  loss  1.380314441652466 correct 50
Time per epoch: 0.1387s Epoch  440  loss  0.5723669711039072 correct 48
Time per epoch: 0.1424s Epoch  450  loss  1.2757781927432397 correct 50
Time per epoch: 0.1402s Epoch  460  loss  0.45290223666765167 correct 49
Time per epoch: 0.1404s Epoch  470  loss  0.31061951443207964 correct 50
Time per epoch: 0.1420s Epoch  480  loss  0.6078854825417415 correct 49
Time per epoch: 0.1423s Epoch  490  loss  0.6662814637563936 correct 49
