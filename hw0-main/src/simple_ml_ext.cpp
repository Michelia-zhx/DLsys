#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


// def data_iter(features, labels, batch_size):
//     num_examples = len(features)
//     indices = list(range(num_examples))
//     for i in range(0, num_examples, batch_size):
//         j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
//         yield features[j], labels[j]

// def softmax(X):
//         X_exp = np.exp(X)
//         partition = X_exp.sum(axis=1, keepdims=True)
//         return X_exp / partition

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of exmaples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    for (int i=0; i<m; i+= batch){
        int j = std::min(i+batch, m);
        int batch_size = j-i;
        float *logits = new float[batch_size*k];
        float *grad = new float[n*k];
        for (int l=0; l<batch_size; l++){
            for (int c=0; c<k; c++){
                logits[l*k+c] = 0;
                for (int d=0; d<n; d++){
                    logits[l*k+c] += X[(i+l)*n+d] * theta[d*k+c];
                }
            }
        }
        for (int l=0; l<batch_size; l++){
            float sum = 0;
            for (int c=0; c<k; c++){
                logits[l*k+c] = std::exp(logits[l*k+c]);
                sum += logits[l*k+c];
            }
            for (int c=0; c<k; c++){
                logits[l*k+c] /= sum;
            }
        }
        for (int l=0; l<batch_size; l++){
            for (int c=0; c<k; c++){
                logits[l*k+c] -= (c==y[i+l]);
            }
        }
        for (int d=0; d<n; d++){
            for (int c=0; c<k; c++){
                grad[d*k+c] = 0;
                for (int l=0; l<batch_size; l++){
                    grad[d*k+c] += logits[l*k+c]*X[(i+l)*n+d];
                }
            }
        }
        for (int d=0; d<n; d++){
            for (int c=0; c<k; c++){
                theta[d*k+c] -= lr*grad[d*k+c]/batch_size;
            }
        }
        
        delete[] logits;
        delete[] grad;
    }
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
