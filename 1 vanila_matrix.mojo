import benchmark
from random import rand

struct Matrix:

    var height : Int
    var width : Int
    var total_items : Int
    var data: Pointer[Float32]

    fn __init__(inout self, default_value: Float32, height: Int, width : Int) -> None:
        self.height = height if height > 0 else 1
        self.width = width if width > 0 else 1
        self.total_items = height * width
        self.data = Pointer[Float32].alloc(self.total_items)
        for i in range(self.total_items):
            self.data.store(i, default_value)

    fn __getitem__(borrowed self, row : Int, column: Int) -> Float32:
        var loc : Int = (row * self.width) + column
        if loc > self.total_items:
            print("You are accessing a value outside the matrix")
            return self.data.load(0)
        
        return self.data.load(loc)

    fn __setitem__(inout self, row : Int, column : Int, item : Float32) -> None:
        var loc : Int = (row * self.width) + column
        if loc > self.total_items:
            print("You are settign a value outside the matrix")
        
        else: self.data.store(loc, item)

    fn __del__(owned self):
        self.data.free()

    fn __len__(borrowed self) -> Int:
        return self.total_items

    fn __copyinit__(inout self, Other : Self) -> None:
        self.height = Other.height
        self.width = Other.width
        self.total_items = Other.total_items
        # we have to use new set of memory since we are copying
        self.data = Pointer[Float32].alloc(self.total_items)
        memcpy(self.data, Other.data, self.total_items)

    fn __eq__(borrowed self, rhs: Matrix) -> Bool:
        for i in range(self.height):
            for j in range(self.width):
                var self_val = self[i, j]

                var rhs_val = rhs[i, j]
                if self_val < rhs_val or self_val > rhs_val:
                    return False

        return True


    fn __add__(borrowed self, rhs: Matrix) -> Matrix:
        if self.width != rhs.width or self.height != rhs.height:
            print("Matrix dimensions do not match")
            return Matrix(0.0, 1, 1)
        var result = Matrix(0.0, self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                result[i, j] = self[i, j] + rhs[i, j]
        
        return result

    fn __mul__(borrowed self, rhs: Matrix) -> Matrix:
        if self.width != rhs.height:
            print("Cannot multply: dimensions")
            return Matrix(0.0, self.height, self.width)
        
        var new_matrix = Matrix(0.0, self.height, rhs.width)
        for i in range(self.height):
            for j in range(rhs.width):
                for k in range(self.width):
                    new_matrix[i, j] += self[i, j] + rhs[k, j]

        return new_matrix
    
    fn __ne__(borrowed self, rhs : Matrix) -> Bool:
        return not self == rhs

    fn apply_function[func : fn(Float32) -> Float32](borrowed self) -> Matrix:
        var new_matrix = Matrix(0.0, self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = func(self[i, j])
        
        return new_matrix


    fn rand(borrowed self) -> None:
        rand(self.data, self.height * self.width)



@always_inline
fn bench():
    var A = Matrix(0.0, 1024, 1024)
    A.rand()
    var B = Matrix(0.0, 1024, 1024)
    B.rand()

    @always_inline
    @parameter
    fn test_fn():
        A * B


    var secs = benchmark.run[test_fn](max_runtime_secs=1).mean()

    var gflops = ((2 * 1024 * 1024 * 1024) / secs) / 1e9

    print(gflops, "GFLOP/s")



fn main() -> None:
    bench()