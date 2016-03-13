#include <iostream>
#include <omp.h>
#include <fstream>
#include <vector>
#include <ctime>

using namespace std;

void matrix_sub_block_mult(vector<vector<int>> &res, vector<vector<int>>left, vector<vector<int>> right, int r, int i_out, int j_out, int k_out) {
    for (int i = i_out * r; i < (i_out + 1) * r; ++i) {
        for (int j = j_out * r; j < (j_out + 1) * r; ++j) {
            for (int k = k_out * r; k < (k_out + 1) * r; ++k) {
                res[i][j] += left[i][k] * right[k][j];
            }
        }
    }
}

void matrix_sub_block_mult_omp(vector<vector<int>> res, vector<vector<int>>left, vector<vector<int>> right, int r, int i_out, int j_out, int k_out) {
    #pragma omp parallel for
    for (int i = i_out * r; i < (i_out + 1) * r; ++i) {
        #pragma omp parallel for
        for (int j = j_out * r; j < (j_out + 1) * r; ++j) {
            for (int k = k_out * r; k < (k_out + 1) * r; ++k) {
                res[i][j] += left[i][k] * right[k][j];
            }
        }
    }
}

vector<vector<int>> matrix_simple_mult(vector<vector<int>> left, vector<vector<int>> right, int n1, int n3, int n2) {
    vector<vector<int>> res(n1, vector<int>(n2, 0));

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            for (int k = 0; k < n3; ++k) {
                res[i][j] += left[i][k] * right[k][j];
            }
        }
    }

    return res;
}

vector<vector<int>> matrix_simple_mult_omp(vector<vector<int>> left, vector<vector<int>> right, int n1, int n3, int n2) {
    vector<vector<int>> res(n1, vector<int>(n2, 0));
    #pragma omp parallel for
    for (int i = 0; i < n1; ++i) {
        #pragma omp parallel for
        for (int j = 0; j < n2; ++j) {
            for (int k = 0; k < n3; ++k) {
                res[i][j] += left[i][k] * right[k][j];
            }
        }
    }
    return res;
}

vector<vector<int>> matrix_block_mult(vector<vector<int>>left, vector<vector<int>>right, int n1, int n3, int n2, int r) {
    int q1 = n1 / r;
    int q2 = n2 / r;
    int q3 = n3 / r;
    vector<vector<int>> res(n1, vector<int>(n2, 0));

    for (int i = 0; i < q1; ++i) {
        for (int j = 0; j < q2; ++j) {
            for (int k = 0; k < q3; ++k) {
                matrix_sub_block_mult(res, left, right, r, i, j, k);
            }
        }
    }
    return res;
}

vector<vector<int>> matrix_block_mult_external_omp(vector<vector<int>>left, vector<vector<int>>right, int n1, int n3, int n2, int r) {
    int q1 = n1 / r;
    int q2 = n2 / r;
    int q3 = n3 / r;
    vector<vector<int>> res(n1, vector<int>(n2, 0));

    #pragma omp parallel for
    for (int i = 0; i < q1; ++i) {
        #pragma omp parallel for
        for (int j = 0; j < q2; ++j) {
            for (int k = 0; k < q3; ++k) {
                matrix_sub_block_mult(res, left, right, r, i, j, k);
            }
        }
    }
    return res;
}

vector<vector<int>> matrix_block_mult_internal_omp(vector<vector<int>>left, vector<vector<int>>right, int n1, int n3, int n2, int r) {
    int q1 = n1 / r;
    int q2 = n2 / r;
    int q3 = n3 / r;
    vector<vector<int>> res(n1, vector<int>(n2, 0));

    for (int i = 0; i < q1; ++i) {
        for (int j = 0; j < q2; ++j) {
            for (int k = 0; k < q3; ++k) {
                matrix_sub_block_mult_omp(res, left, right, r, i, j, k);
            }
        }
    }
    return res;
}

vector<vector<int>> create_random_matrix(int n, int m){
    vector<vector<int>> random_matrix(n, vector<int>(m));
    for (int i = 0; i < random_matrix.size(); i++) {
        for (int j = 0; j < random_matrix[i].size(); j++) {
            random_matrix[i][j] = rand() % 101 - 100;
        }
    }
    return random_matrix;
}

int main() {

    srand(time(0));

    ofstream output;
    output.open("output.csv");
    output.precision(2);
    int N = 500;
    int n1 = 0, n2 = 0, n3 = 0, r = 20;

    vector<vector<int>> left;
    vector<vector<int>> right;

    double init_time;

    output << "N Point Point_omp Block Block_external Block_internal" << endl;

    for (int i = 0; i <= N; i += 100){
        cout << i << endl;
        n1 = n2 = n3 = i;
        left = create_random_matrix(n1, n3);
        right = create_random_matrix(n3, n2);
        output << fixed << i << " ";

        init_time = omp_get_wtime();
        matrix_simple_mult(left, right, n1, n3, n2);
        output << fixed << omp_get_wtime() - init_time << " ";

        init_time = omp_get_wtime();
        matrix_simple_mult_omp(left, right, n1, n3, n2);
        output << fixed << omp_get_wtime() - init_time << " ";

        init_time = omp_get_wtime();
        matrix_block_mult(left, right, n1, n3, n2, r);
        output << fixed << omp_get_wtime() - init_time << " ";

        init_time = omp_get_wtime();
        matrix_block_mult_external_omp(left, right, n1, n3, n2, r);
        output << fixed << omp_get_wtime() - init_time << " ";

        init_time = omp_get_wtime();
        matrix_block_mult_internal_omp(left, right, n1, n3, n2, r);
        output << fixed << omp_get_wtime() - init_time << endl;
    }

    output << endl << "Fix N = 500, change r:" << endl;

    output << "r Block_external Block_internal" << endl;

    N = 500;
    n1 = n2 = n3 = N;
    left = create_random_matrix(n1, n3);
    right = create_random_matrix(n3, n2);
    int rs[] = {10, 20, 40, 50, 100};
    for (int i = 0; i < 5; i ++){
        cout << i << endl;

        output << fixed << rs[i] << " ";

        init_time = omp_get_wtime();
        matrix_block_mult(left, right, n1, n3, n2, rs[i]);
        output << fixed << omp_get_wtime() - init_time << " ";

        init_time = omp_get_wtime();
        matrix_block_mult_external_omp(left, right, n1, n3, n2, rs[i]);
        output << fixed << omp_get_wtime() - init_time << " ";

        init_time = omp_get_wtime();
        matrix_block_mult_internal_omp(left, right, n1, n3, n2, rs[i]);
        output << fixed << omp_get_wtime() - init_time << endl;
    }
}