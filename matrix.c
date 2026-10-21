#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define EPSILON 1e-9

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

/* Memory Management & Utilities */
Matrix* create_matrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;
    m->rows = rows;
    m->cols = cols;
    m->data = (double**)malloc(rows * sizeof(double*));
    if (!m->data) {
        free(m);
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        m->data[i] = (double*)calloc(cols, sizeof(double));
        if (!m->data[i]) {
            for (int j = 0; j < i; j++) free(m->data[j]);
            free(m->data);
            free(m);
            return NULL;
        }
    }
    return m;
}

void free_matrix(Matrix *m) {
    if (!m) return;
    if (m->data) {
        for (int i = 0; i < m->rows; i++) {
            if (m->data[i]) free(m->data[i]);
        }
        free(m->data);
    }
    free(m);
}

void print_matrix(const char *name, const Matrix *m) {
    if (!m) {
        printf("%s: (NULL)\n", name);
        return;
    }
    printf("%s (%dx%d):\n", name, m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("  [ ");
        for (int j = 0; j < m->cols; j++) {
            double val = m->data[i][j];
            if (fabs(val) < 1e-9) val = 0.0;
            printf("%8.4f ", val);
        }
        printf("]\n");
    }
    printf("\n");
}

Matrix* copy_matrix(const Matrix *m) {
    if (!m) return NULL;
    Matrix *res = create_matrix(m->rows, m->cols);
    if (!res) return NULL;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            res->data[i][j] = m->data[i][j];
        }
    }
    return res;
}

/* Declarations for operations */
Matrix* matrix_add(const Matrix *a, const Matrix *b);
Matrix* matrix_subtract(const Matrix *a, const Matrix *b);
Matrix* matrix_scalar_multiply(const Matrix *a, double scalar);
Matrix* matrix_multiply(const Matrix *a, const Matrix *b);
Matrix* matrix_hadamard(const Matrix *a, const Matrix *b);
Matrix* matrix_transpose(const Matrix *m);

double matrix_trace(const Matrix *m);
double matrix_determinant(const Matrix *m);
int matrix_rank(const Matrix *m);
double matrix_frobenius_norm(const Matrix *m);
double matrix_infinity_norm(const Matrix *m);

bool is_matrix_symmetric(const Matrix *m);
bool is_matrix_skew_symmetric(const Matrix *m);
bool is_matrix_diagonal(const Matrix *m);
bool is_matrix_identity(const Matrix *m);

Matrix* matrix_inverse(const Matrix *m);
Matrix* matrix_power(const Matrix *m, int p);
Matrix* matrix_kronecker(const Matrix *a, const Matrix *b);
bool matrix_lu_decomposition(const Matrix *m, Matrix **L, Matrix **U);
bool matrix_qr_decomposition(const Matrix *m, Matrix **Q, Matrix **R);
bool matrix_eigen_jacobi(const Matrix *m, Matrix **eigenvalues_diag, Matrix **eigenvectors, int max_iter);

void run_all_operations_demo(void);

int main(void) {
    run_all_operations_demo();
    return 0;
}

/* Basic Matrix Arithmetic Operations */

Matrix* matrix_add(const Matrix *a, const Matrix *b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    Matrix *res = create_matrix(a->rows, a->cols);
    if (!res) return NULL;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            res->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
    return res;
}

Matrix* matrix_subtract(const Matrix *a, const Matrix *b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    Matrix *res = create_matrix(a->rows, a->cols);
    if (!res) return NULL;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            res->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    return res;
}

Matrix* matrix_scalar_multiply(const Matrix *a, double scalar) {
    if (!a) return NULL;
    Matrix *res = create_matrix(a->rows, a->cols);
    if (!res) return NULL;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            res->data[i][j] = a->data[i][j] * scalar;
        }
    }
    return res;
}

Matrix* matrix_multiply(const Matrix *a, const Matrix *b) {
    if (!a || !b || a->cols != b->rows) return NULL;
    Matrix *res = create_matrix(a->rows, b->cols);
    if (!res) return NULL;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i][k] * b->data[k][j];
            }
            res->data[i][j] = sum;
        }
    }
    return res;
}

Matrix* matrix_hadamard(const Matrix *a, const Matrix *b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    Matrix *res = create_matrix(a->rows, a->cols);
    if (!res) return NULL;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            res->data[i][j] = a->data[i][j] * b->data[i][j];
        }
    }
    return res;
}

Matrix* matrix_transpose(const Matrix *m) {
    if (!m) return NULL;
    Matrix *res = create_matrix(m->cols, m->rows);
    if (!res) return NULL;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            res->data[j][i] = m->data[i][j];
        }
    }
    return res;
}

/* Matrix Properties and Scalar Attributes */

double matrix_trace(const Matrix *m) {
    if (!m || m->rows != m->cols) return 0.0;
    double trace = 0.0;
    for (int i = 0; i < m->rows; i++) {
        trace += m->data[i][i];
    }
    return trace;
}

double matrix_determinant(const Matrix *m) {
    if (!m || m->rows != m->cols) return 0.0;
    int n = m->rows;
    Matrix *temp = copy_matrix(m);
    if (!temp) return 0.0;

    double det = 1.0;
    int swaps = 0;

    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(temp->data[k][i]) > fabs(temp->data[pivot][i])) {
                pivot = k;
            }
        }

        if (pivot != i) {
            double *tmp_row = temp->data[i];
            temp->data[i] = temp->data[pivot];
            temp->data[pivot] = tmp_row;
            swaps++;
        }

        if (fabs(temp->data[i][i]) < EPSILON) {
            free_matrix(temp);
            return 0.0;
        }

        det *= temp->data[i][i];

        for (int k = i + 1; k < n; k++) {
            double factor = temp->data[k][i] / temp->data[i][i];
            for (int j = i + 1; j < n; j++) {
                temp->data[k][j] -= factor * temp->data[i][j];
            }
        }
    }

    if (swaps % 2 != 0) det = -det;

    free_matrix(temp);
    return det;
}

int matrix_rank(const Matrix *m) {
    if (!m) return 0;
    Matrix *temp = copy_matrix(m);
    if (!temp) return 0;

    int rows = temp->rows;
    int cols = temp->cols;
    int rank = 0;

    bool row_selected[rows];
    for (int i = 0; i < rows; i++) row_selected[i] = false;

    for (int j = 0; j < cols; j++) {
        int pivot_row = -1;
        double max_val = EPSILON;

        for (int i = 0; i < rows; i++) {
            if (!row_selected[i] && fabs(temp->data[i][j]) > max_val) {
                max_val = fabs(temp->data[i][j]);
                pivot_row = i;
            }
        }

        if (pivot_row != -1) {
            rank++;
            row_selected[pivot_row] = true;

            for (int i = 0; i < rows; i++) {
                if (i != pivot_row && fabs(temp->data[i][j]) > EPSILON) {
                    double factor = temp->data[i][j] / temp->data[pivot_row][j];
                    for (int k = j; k < cols; k++) {
                        temp->data[i][k] -= factor * temp->data[pivot_row][k];
                    }
                }
            }
        }
    }

    free_matrix(temp);
    return rank;
}

double matrix_frobenius_norm(const Matrix *m) {
    if (!m) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            sum += m->data[i][j] * m->data[i][j];
        }
    }
    return sqrt(sum);
}

double matrix_infinity_norm(const Matrix *m) {
    if (!m) return 0.0;
    double max_sum = 0.0;
    for (int i = 0; i < m->rows; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < m->cols; j++) {
            row_sum += fabs(m->data[i][j]);
        }
        if (row_sum > max_sum) max_sum = row_sum;
    }
    return max_sum;
}

bool is_matrix_symmetric(const Matrix *m) {
    if (!m || m->rows != m->cols) return false;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (fabs(m->data[i][j] - m->data[j][i]) > EPSILON) return false;
        }
    }
    return true;
}

bool is_matrix_skew_symmetric(const Matrix *m) {
    if (!m || m->rows != m->cols) return false;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (fabs(m->data[i][j] + m->data[j][i]) > EPSILON) return false;
        }
    }
    return true;
}

bool is_matrix_diagonal(const Matrix *m) {
    if (!m) return false;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (i != j && fabs(m->data[i][j]) > EPSILON) return false;
        }
    }
    return true;
}

bool is_matrix_identity(const Matrix *m) {
    if (!m || m->rows != m->cols) return false;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (i == j) {
                if (fabs(m->data[i][j] - 1.0) > EPSILON) return false;
            } else {
                if (fabs(m->data[i][j]) > EPSILON) return false;
            }
        }
    }
    return true;
}

/* Advanced Transformations and Decompositions */

Matrix* matrix_inverse(const Matrix *m) {
    if (!m || m->rows != m->cols) return NULL;
    int n = m->rows;

    Matrix *aug = create_matrix(n, 2 * n);
    if (!aug) return NULL;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug->data[i][j] = m->data[i][j];
        }
        aug->data[i][n + i] = 1.0;
    }

    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(aug->data[k][i]) > fabs(aug->data[pivot][i])) {
                pivot = k;
            }
        }

        if (pivot != i) {
            double *tmp = aug->data[i];
            aug->data[i] = aug->data[pivot];
            aug->data[pivot] = tmp;
        }

        double pivot_val = aug->data[i][i];
        if (fabs(pivot_val) < EPSILON) {
            free_matrix(aug);
            return NULL; /* Matrix is singular */
        }

        for (int j = 0; j < 2 * n; j++) {
            aug->data[i][j] /= pivot_val;
        }

        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = aug->data[k][i];
                for (int j = 0; j < 2 * n; j++) {
                    aug->data[k][j] -= factor * aug->data[i][j];
                }
            }
        }
    }

    Matrix *inv = create_matrix(n, n);
    if (!inv) {
        free_matrix(aug);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv->data[i][j] = aug->data[i][n + j];
        }
    }

    free_matrix(aug);
    return inv;
}

Matrix* matrix_power(const Matrix *m, int p) {
    if (!m || m->rows != m->cols || p < 0) return NULL;
    int n = m->rows;

    Matrix *res = create_matrix(n, n);
    if (!res) return NULL;
    for (int i = 0; i < n; i++) res->data[i][i] = 1.0;

    if (p == 0) return res;

    Matrix *base = copy_matrix(m);
    if (!base) {
        free_matrix(res);
        return NULL;
    }

    while (p > 0) {
        if (p % 2 == 1) {
            Matrix *next_res = matrix_multiply(res, base);
            free_matrix(res);
            res = next_res;
        }
        Matrix *next_base = matrix_multiply(base, base);
        free_matrix(base);
        base = next_base;
        p /= 2;
    }

    free_matrix(base);
    return res;
}

Matrix* matrix_kronecker(const Matrix *a, const Matrix *b) {
    if (!a || !b) return NULL;
    Matrix *res = create_matrix(a->rows * b->rows, a->cols * b->cols);
    if (!res) return NULL;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            for (int k = 0; k < b->rows; k++) {
                for (int l = 0; l < b->cols; l++) {
                    res->data[i * b->rows + k][j * b->cols + l] = a->data[i][j] * b->data[k][l];
                }
            }
        }
    }
    return res;
}

bool matrix_lu_decomposition(const Matrix *m, Matrix **L, Matrix **U) {
    if (!m || m->rows != m->cols || !L || !U) return false;
    int n = m->rows;

    *L = create_matrix(n, n);
    *U = create_matrix(n, n);
    if (!*L || !*U) return false;

    for (int i = 0; i < n; i++) {
        for (int k = i; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += ((*L)->data[i][j] * (*U)->data[j][k]);
            }
            (*U)->data[i][k] = m->data[i][k] - sum;
        }

        for (int k = i; k < n; k++) {
            if (i == k) {
                (*L)->data[i][i] = 1.0;
            } else {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += ((*L)->data[k][j] * (*U)->data[j][i]);
                }
                if (fabs((*U)->data[i][i]) < EPSILON) return false;
                (*L)->data[k][i] = (m->data[k][i] - sum) / (*U)->data[i][i];
            }
        }
    }
    return true;
}

bool matrix_qr_decomposition(const Matrix *m, Matrix **Q, Matrix **R) {
    if (!m || !Q || !R) return false;
    int rows = m->rows;
    int cols = m->cols;

    *Q = create_matrix(rows, cols);
    *R = create_matrix(cols, cols);
    if (!*Q || !*R) return false;

    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            (*Q)->data[i][j] = m->data[i][j];
        }

        for (int i = 0; i < j; i++) {
            double dot = 0.0;
            for (int k = 0; k < rows; k++) {
                dot += m->data[k][j] * (*Q)->data[k][i];
            }
            (*R)->data[i][j] = dot;
            for (int k = 0; k < rows; k++) {
                (*Q)->data[k][j] -= dot * (*Q)->data[k][i];
            }
        }

        double norm = 0.0;
        for (int k = 0; k < rows; k++) {
            norm += (*Q)->data[k][j] * (*Q)->data[k][j];
        }
        norm = sqrt(norm);

        (*R)->data[j][j] = norm;
        if (norm > EPSILON) {
            for (int k = 0; k < rows; k++) {
                (*Q)->data[k][j] /= norm;
            }
        }
    }
    return true;
}

bool matrix_eigen_jacobi(const Matrix *m, Matrix **eigenvalues_diag, Matrix **eigenvectors, int max_iter) {
    if (!m || m->rows != m->cols || !is_matrix_symmetric(m)) return false;
    int n = m->rows;

    Matrix *A = copy_matrix(m);
    Matrix *V = create_matrix(n, n);
    if (!A || !V) {
        free_matrix(A);
        free_matrix(V);
        return false;
    }
    for (int i = 0; i < n; i++) V->data[i][i] = 1.0;

    for (int iter = 0; iter < max_iter; iter++) {
        int p = 0, q = 1;
        double max_offdiag = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (fabs(A->data[i][j]) > max_offdiag) {
                    max_offdiag = fabs(A->data[i][j]);
                    p = i;
                    q = j;
                }
            }
        }

        if (max_offdiag < EPSILON) break;

        double app = A->data[p][p];
        double aqq = A->data[q][q];
        double apq = A->data[p][q];

        double theta = 0.5 * atan2(2.0 * apq, aqq - app);
        double c = cos(theta);
        double s = sin(theta);

        for (int i = 0; i < n; i++) {
            if (i != p && i != q) {
                double aip = A->data[i][p];
                double aiq = A->data[i][q];
                A->data[i][p] = A->data[p][i] = c * aip - s * aiq;
                A->data[i][q] = A->data[q][i] = s * aip + c * aiq;
            }
            double vip = V->data[i][p];
            double viq = V->data[i][q];
            V->data[i][p] = c * vip - s * viq;
            V->data[i][q] = s * vip + c * viq;
        }

        A->data[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A->data[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A->data[p][q] = A->data[q][p] = 0.0;
    }

    *eigenvalues_diag = A;
    *eigenvectors = V;
    return true;
}

void run_all_operations_demo(void) {
    printf("====================================================\n");
    printf("     MATRIX OPERATIONS IN THE UNIVERSE DEMO         \n");
    printf("====================================================\n\n");

    /* 1. Basic Creation and Printing */
    Matrix *A = create_matrix(3, 3);
    A->data[0][0] = 4; A->data[0][1] = 1; A->data[0][2] = -1;
    A->data[1][0] = 1; A->data[1][1] = 2; A->data[1][2] = 1;
    A->data[2][0] = -1; A->data[2][1] = 1; A->data[2][2] = 5;

    Matrix *B = create_matrix(3, 3);
    B->data[0][0] = 1; B->data[0][1] = 0; B->data[0][2] = 2;
    B->data[1][0] = -1; B->data[1][1] = 3; B->data[1][2] = 1;
    B->data[2][0] = 4; B->data[2][1] = 0; B->data[2][2] = 2;

    print_matrix("Matrix A", A);
    print_matrix("Matrix B", B);

    /* 2. Basic Arithmetic */
    Matrix *C_add = matrix_add(A, B);
    print_matrix("A + B", C_add);
    free_matrix(C_add);

    Matrix *C_sub = matrix_subtract(A, B);
    print_matrix("A - B", C_sub);
    free_matrix(C_sub);

    Matrix *C_scalar = matrix_scalar_multiply(A, 2.5);
    print_matrix("2.5 * A", C_scalar);
    free_matrix(C_scalar);

    Matrix *C_mul = matrix_multiply(A, B);
    print_matrix("A * B (Matrix Multiplication)", C_mul);
    free_matrix(C_mul);

    Matrix *C_had = matrix_hadamard(A, B);
    print_matrix("A o B (Hadamard Product)", C_had);
    free_matrix(C_had);

    Matrix *C_trans = matrix_transpose(B);
    print_matrix("B^T (Transpose of B)", C_trans);
    free_matrix(C_trans);

    /* 3. Properties and Scalar Values */
    printf("--- Matrix Properties ---\n");
    printf("Trace(A): %.4f\n", matrix_trace(A));
    printf("Determinant(A): %.4f\n", matrix_determinant(A));
    printf("Rank(A): %d\n", matrix_rank(A));
    printf("Frobenius Norm(A): %.4f\n", matrix_frobenius_norm(A));
    printf("Infinity Norm(A): %.4f\n", matrix_infinity_norm(A));
    printf("Is A Symmetric? %s\n", is_matrix_symmetric(A) ? "Yes" : "No");
    printf("Is A Skew-Symmetric? %s\n", is_matrix_skew_symmetric(A) ? "Yes" : "No");
    printf("Is A Diagonal? %s\n", is_matrix_diagonal(A) ? "Yes" : "No");
    printf("Is A Identity? %s\n\n", is_matrix_identity(A) ? "Yes" : "No");

    /* 4. Inverse & Power */
    Matrix *A_inv = matrix_inverse(A);
    print_matrix("A^-1 (Inverse of A)", A_inv);
    if (A_inv) {
        Matrix *A_Ainv = matrix_multiply(A, A_inv);
        print_matrix("A * A^-1 (Identity check)", A_Ainv);
        free_matrix(A_Ainv);
        free_matrix(A_inv);
    }

    Matrix *A_pow3 = matrix_power(A, 3);
    print_matrix("A^3 (Matrix Power 3)", A_pow3);
    free_matrix(A_pow3);

    /* 5. Kronecker Product */
    Matrix *K1 = create_matrix(2, 2);
    K1->data[0][0] = 1; K1->data[0][1] = 2;
    K1->data[1][0] = 3; K1->data[1][1] = 4;

    Matrix *K2 = create_matrix(2, 2);
    K2->data[0][0] = 0; K2->data[0][1] = 5;
    K2->data[1][0] = 6; K2->data[1][1] = 7;

    Matrix *Kron = matrix_kronecker(K1, K2);
    print_matrix("Kronecker Product (K1 (x) K2)", Kron);
    free_matrix(K1);
    free_matrix(K2);
    free_matrix(Kron);

    /* 6. LU Decomposition */
    Matrix *L = NULL, *U = NULL;
    if (matrix_lu_decomposition(A, &L, &U)) {
        print_matrix("LU Decomposition: L", L);
        print_matrix("LU Decomposition: U", U);
        Matrix *LU = matrix_multiply(L, U);
        print_matrix("L * U (Reconstructing A)", LU);
        free_matrix(LU);
        free_matrix(L);
        free_matrix(U);
    }

    /* 7. QR Decomposition */
    Matrix *Q = NULL, *R = NULL;
    if (matrix_qr_decomposition(A, &Q, &R)) {
        print_matrix("QR Decomposition: Q", Q);
        print_matrix("QR Decomposition: R", R);
        Matrix *QR = matrix_multiply(Q, R);
        print_matrix("Q * R (Reconstructing A)", QR);
        free_matrix(QR);
        free_matrix(Q);
        free_matrix(R);
    }

    /* 8. Eigenvalues and Eigenvectors (Jacobi Method for Symmetric A) */
    Matrix *eigenvalues = NULL, *eigenvectors = NULL;
    if (matrix_eigen_jacobi(A, &eigenvalues, &eigenvectors, 100)) {
        print_matrix("Eigenvalues (Diagonal)", eigenvalues);
        print_matrix("Eigenvectors (Columns)", eigenvectors);
        free_matrix(eigenvalues);
        free_matrix(eigenvectors);
    }

    free_matrix(A);
    free_matrix(B);
    printf("====================================================\n");
    printf("          DEMO COMPLETED SUCCESSFULLY               \n");
    printf("====================================================\n");
}
