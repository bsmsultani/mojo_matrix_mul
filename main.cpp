#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <functional>
#include <algorithm>
#include <mutex>

using namespace std;

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;
constexpr int tile_size = 4;
constexpr int simd_width = 4 * 2;


template<typename T>
class Matrix {
public:
    int rows, cols;
    vector<T> data;

    Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols) {
        fill(data.begin(), data.end(), 0);
    }

    Matrix(int rows, int cols, T* external_data) : rows(rows), cols(cols), data(external_data, external_data + rows * cols) {}

    static Matrix<T> random(int rows, int cols) {
        Matrix<T> mat(rows, cols);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0, 1);
        for (auto &element : mat.data) {
            element = dis(gen);
        }
        return mat;
    }

    T &operator()(int row, int col) {
        return data[row * cols + col];
    }

    const T &operator()(int row, int col) const {
        return data[row * cols + col];
    }
};

void matmul_naive(Matrix<float> &C, const Matrix<float> &A, const Matrix<float> &B) {
    for (int m = 0; m < C.rows; ++m) {
        for (int k = 0; k < A.cols; ++k) {
            for (int n = 0; n < C.cols; ++n) {
                C(m, n) += A(m, k) * B(k, n);
            }
        }
    }
}

void matmul_vectorized(Matrix<float> &C, const Matrix<float> &A, const Matrix<float> &B) {
    for (int m = 0; m < C.rows; ++m) {
        for (int k = 0; k < A.cols; ++k) {
            for (int n = 0; n <= C.cols - simd_width; n += simd_width) {
                for (int i = 0; i < simd_width; ++i) {
                    C(m, n + i) += A(m, k) * B(k, n + i);
                }
            }
            for (int n = (C.cols / simd_width) * simd_width; n < C.cols; ++n) {
                C(m, n) += A(m, k) * B(k, n);
            }
        }
    }
}

void matmul_parallelized(Matrix<float> &C, const Matrix<float> &A, const Matrix<float> &B) {
    auto worker = [&](int start_row, int end_row) {
        for (int m = start_row; m < end_row; ++m) {
            for (int k = 0; k < A.cols; ++k) {
                for (int n = 0; n <= C.cols - simd_width; n += simd_width) {
                    for (int i = 0; i < simd_width; ++i) {
                        C(m, n + i) += A(m, k) * B(k, n + i);
                    }
                }
                for (int n = (C.cols / simd_width) * simd_width; n < C.cols; ++n) {
                    C(m, n) += A(m, k) * B(k, n);
                }
            }
        }
    };

    int num_threads = thread::hardware_concurrency();

    vector<thread> threads;
    int rows_per_thread = C.rows / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? C.rows : start_row + rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
    }

    for (auto &t : threads) {
        t.join();
    }
}

void matmul_tiled_parallelized(Matrix<float> &C, const Matrix<float> &A, const Matrix<float> &B) {
    auto worker = [&](int start_row, int end_row) {
        for (int m = start_row; m < end_row; ++m) {
            for (int y = 0; y < A.cols; y += tile_size) {
                for (int x = 0; x < C.cols; x += tile_size) {
                    for (int k = y; k < y + tile_size && k < A.cols; ++k) {
                        for (int n = x; n < x + tile_size && n < C.cols; ++n) {
                            C(m, n) += A(m, k) * B(k, n);
                        }
                    }
                }
            }
        }
    };

    int num_threads = thread::hardware_concurrency();
    vector<thread> threads;
    int rows_per_thread = C.rows / num_threads;


    for (int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? C.rows : start_row + rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
    }

    for (auto &t : threads) {
        t.join();
    }
}

template<typename Func>
void benchmark(Func func, double base_gflops) {
    Matrix<float> C(M, N);
    Matrix<float> A = Matrix<float>::random(M, K);
    Matrix<float> B = Matrix<float>::random(K, N);

    Matrix<int> D = Matrix<int>::random(K, N);

    auto start = chrono::high_resolution_clock::now();
    func(C, A, B);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    double secs = elapsed.count();
    double gflops = (2.0 * M * N * K) / (secs * 1e9);
    double speedup = gflops / base_gflops;

    cout << gflops << " GFLOP/s, a " << speedup << "x speedup over Vanilla Implementation!" << endl;
}

int main() {
    double vanilla_gflop = 0.277533;

    cout << "Running the vanilla implementation!" << endl;
    benchmark(matmul_naive, vanilla_gflop);

    cout << "Running the vectorized implementation!" << endl;
    benchmark(matmul_vectorized, vanilla_gflop);

    cout << "Running the parallelized implementation!" << endl;
    benchmark(matmul_parallelized, vanilla_gflop);

    cout << "Running tiled parallelized implementation!" << endl;
    benchmark(matmul_tiled_parallelized, vanilla_gflop);

    return 0;
}
