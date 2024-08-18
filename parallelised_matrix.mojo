import benchmark
from memory import memset_zero
from random import rand, random_float64

alias type = DType.float32


# rows can also be accessed as self.rows as well
struct Matrix[rows: Int, cols: Int]:
    # pointer holding the data for the matrix
    var data: DTypePointer[type]

    # Initialize zeroeing all values
    fn __init__(inout self):

        # DTypePointer is specialised for storing and loading SIMD values
        self.data = DTypePointer[type].alloc(rows * cols)
        # setting the values in memory to zero
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: DTypePointer[type]):
        self.data = data

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    # loads data from the memory
    #   - width parameter species the number of elements to load from memory
    #   - we only want one element at position y * self.cols + x to load
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)



    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)


# Note that C, A, and B have types.
fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


# [1, 2]   [5, 6]   [1 * 5 + 2 * 7]
# [3, 4] x [7, 8] = [1 * 6 + 2 * 8]

# 1st Iteration: C[0, 0] = A[0, 0] * B[0, 0]
# 2nd Iteration: C[0 ,1] = A[0, 0] * B[0, 1]

# 3rd Iteration: C[0, 0] = A[0, 1] * B[1, 0]
# 4th Iteration: C[0, 1] = A[0, 1] * B[1, 1]

# 5th Iteration: C[0, 1] ...
# 5th Iteration: C[1, 1] ...


alias M = 1024
alias N = 1024
alias K = 1024


# simdwidthof = number of float32 elements that fit into a single SIMD register
# using a 2x multiplier allows some SIMD operations to run in the same cycle
alias nelts = simdwidthof[DType.float32]() * 2



# Simplify the code by using the builtin vectorize function
from algorithm import vectorize

fn matmul_vectorized_1(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            @parameter
            fn dot[nelts: Int](n: Int):
                C.store(m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n))
            # run each vector operation in the SIMD
            vectorize[dot, nelts, size = C.cols]()





# Parallelize the code by using the builtin parallelize function
from algorithm import parallelize

fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[dot, nelts, size = C.cols]()
    # parallelise each vector operation
    parallelize[calc_row](C.rows, C.rows)




from algorithm import Static2DTileUnitFunc as Tile2DFunc

# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)



# Use the above tile function to perform tiled matmul.
fn matmul_tiled_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(m, n + x, C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x))
                vectorize[dot, nelts, size = tile_x]()

        # We hardcode the tile factor to be 4.
        alias tile_size = 8
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](C.rows, C.rows)




@always_inline
fn bench[
    func: fn (Matrix, Matrix, Matrix) -> None](base_gflops: Float64):
    var C = Matrix[M, N]()
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()

    # decorator is used to force the compiler to 'inline the function'
    # the body of the function is directly inserted to the calling function
    # reduces overhead of of jumping to different location in the code
    
    @always_inline
    # forces the compiler to decide in advance what variables that are used from the outer function
    @parameter
    fn test_fn():
        _ = func(C, A, B)
        
    var secs = benchmark.run[test_fn](max_runtime_secs=1).mean()

    A.data.free()
    B.data.free()
    C.data.free()

    var gflops = ((2 * M * N * K) / secs) / 1e9
    var speedup: Float64 = gflops / base_gflops

    print(gflops, "GFLOP/s, a", speedup, "x speedup over Vanilla Implementation!")




def main():
    alias vanilla_gflop = 6.0371187885375859
    print("Running the vanilla implementation!")
    bench[matmul_naive](vanilla_gflop)
    print("Running the vecrtorised Implementation!")
    bench[matmul_vectorized_1](vanilla_gflop)
    print("Running the parallelised Implementation!")
    bench[matmul_parallelized](vanilla_gflop)
    print("Running tiled parallelised Implementation!")
    bench[matmul_tiled_parallelized](vanilla_gflop)