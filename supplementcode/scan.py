####################### Packages #######################

from .core import *
from sklearn.utils.extmath import _incremental_mean_and_var

####################### Additional Licences #####################
### BSD 3-Clause License
"""
Copyright (c) 2007-2021 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

### pyChemometrics
"""BSD 3-Clause License

Copyright (c) 2017, Gonçalo dos Santos Correia
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

####################### Classes ######################

class CustomScalerAuto(TransformerMixin, BaseEstimator):
    """
    Auto scaler.
    
    Custom scaler class for pipeline. Customized for auto scaling.
    
    Parameters
    ----------
    scale_power : float
        To which power should the standard deviation of each variable be raised for scaling. 
        0: Mean centering; 0.5: Pareto; 1: Unit Variance.
    copy : bool
        Copy the array containing the data.
    with_mean : bool
        Perform mean centering.
    with_std : bool
        Scale the data.
    """

    def __init__(self, *, scale_power=1, copy=True, with_mean=True, with_std=True):
        # Initialization
        self.scale_power = scale_power
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        # Checking attribute, attributes set together
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit transformer.
        
        Compute the mean and standard deviation from a dataset to use in future scaling operations.
                
        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        """
        Partial fit transformer.
        
        Performs online computation of mean and standard deviation on X for later scaling.
        All of X is processed as a single batch.
        This is intended for cases when `fit` is
        not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.

        Notes
        -----
        [1] Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms for computing the sample variance: Analysis and recommendations."The American Statistician 37.3 (1983): 242-247
        """
        first_call = not hasattr(self, "n_samples_seen_")
        # Validate data
        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan', reset=first_call)
        n_features = X.shape[1]

        # Check sample weights
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        # Set data type
        dtype = numpy.int64 if sample_weight is None else X.dtype
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = numpy.zeros(n_features, dtype=dtype)
        elif numpy.size(self.n_samples_seen_) == 1:
            self.n_samples_seen_ = numpy.repeat(
                self.n_samples_seen_, X.shape[1])
            self.n_samples_seen_ = \
                self.n_samples_seen_.astype(dtype, copy=False)

        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            sparse_constructor = (scipy.sparse.csr_matrix
                                  if X.format == 'csr' else scipy.sparse.csc_matrix)

            if self.with_std:
                # First pass
                if not hasattr(self, 'scale_'):
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        mean_variance_axis(X, axis=0, weights=sample_weight,
                                           return_sum_weights=True)
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_,
                                                weights=sample_weight)
                # Set data type
                self.mean_ = self.mean_.astype(numpy.float64, copy=False)
                self.var_ = self.var_.astype(numpy.float64, copy=False)
            else:
                # Must be False for sparse
                self.mean_ = None
                self.var_ = None
                weights = _check_sample_weight(sample_weight, X)
                sum_weights_nan = weights @ sparse_constructor(
                    (numpy.isnan(X.data), X.indices, X.indptr),
                    shape=X.shape)
                self.n_samples_seen_ += (
                    (numpy.sum(weights) - sum_weights_nan).astype(dtype)
                )
        else:
            # First pass
            if not hasattr(self, 'scale_'):
                self.mean_ = .0
                if self.with_std:
                    self.var_ = .0
                else:
                    self.var_ = None

            if not self.with_mean and not self.with_std:
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - numpy.isnan(X).sum(axis=0)

            else:
                self.mean_, self.var_, self.n_samples_seen_ = \
                    _incremental_mean_and_var(X, self.mean_, self.var_,
                                              self.n_samples_seen_)

        if numpy.ptp(self.n_samples_seen_) == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]

        if self.with_std:
            # Extract the list of near constant features on the raw variances,
            # before taking the square root.
            constant_mask = self.var_ < 10 * numpy.finfo(X.dtype).eps
            self.scale_ = _handle_zeros_in_scale(
                numpy.sqrt(self.var_) ** self.scale_power, copy=False, constant_mask=constant_mask)
        else:
            self.scale_ = None

        return self

    def transform(self, X, copy=None):
        """
        Transform data.

        Perform standardization by centering and scaling using the parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        copy : bool
            Copy the X matrix.
        
               
        Returns
        -------
        matrix_scaled : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Scaled version of the matrix.
        """
        # Check if fitted
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        # Validate data
        X = self._validate_data(X, reset=False,
                                accept_sparse='csr', copy=copy,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')
        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X = X - self.mean_
            if self.with_std:
                X = X / self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """
        Inverse transform data.

        Scale back the data to the original representation.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        copy : bool
            Copy the X matrix.
        
        Returns
        -------
        matrix_reverted : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Matrix with the scaling operation reverted.
        """
        # Check if fitted
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy, ensure_2d=False,
                        dtype=FLOAT_DTYPES, force_all_finite="allow-nan")
        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_std:
                X = X * self.scale_
            if self.with_mean:
                X = X + self.mean_
        return X

    def _more_tags(self):
        return {'allow_nan': True,
                'preserves_dtype': [numpy.float64, numpy.float32]}

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class CustomScalerRange(TransformerMixin, BaseEstimator):
    """
    Range scaler.
    
    Custom scaler class for pipeline. Customized for range scaling.
    
    Parameters
    ----------
    copy : bool
        Copy the array containing the data.
    with_mean : bool
        Perform mean centering.
    with_range : bool
        Scale the data.
    """
    def __init__(self, *, copy=True, with_mean=True, with_range=True):
        # Initialization
        self.with_mean = with_mean
        self.with_range = with_range
        self.copy = copy

    def _reset(self):
        # Checking attribute, attributes set together
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_
            del self.range_

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit transformer.
        
        Compute the mean and standard deviation from a dataset to use in future scaling operations.
                
        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        """
        Partial fit transformer.
        
        Performs online computation of mean and standard deviation on X for later scaling.
        All of X is processed as a single batch.
        This is intended for cases when `fit` is
        not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.

        Notes
        -----
        [1] Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms for computing the sample variance: Analysis and recommendations."The American Statistician 37.3 (1983): 242-247
        """
        first_call = not hasattr(self, "n_samples_seen_")
        # Validate data
        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan', reset=first_call)
        n_features = X.shape[1]

        # Check sample weights
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        # Set data type
        dtype = numpy.int64 if sample_weight is None else X.dtype
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = numpy.zeros(n_features, dtype=dtype)
        elif numpy.size(self.n_samples_seen_) == 1:
            self.n_samples_seen_ = numpy.repeat(
                self.n_samples_seen_, X.shape[1])
            self.n_samples_seen_ = \
                self.n_samples_seen_.astype(dtype, copy=False)

        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            sparse_constructor = (scipy.sparse.csr_matrix
                                  if X.format == 'csr' else scipy.sparse.csc_matrix)

            if self.with_range:
                # First pass
                if not hasattr(self, 'scale_'):
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        mean_variance_axis(X, axis=0, weights=sample_weight,
                                           return_sum_weights=True)
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_,
                                                weights=sample_weight)
                    
                    self.range_ = numpy.nanmax(X, axis=0)-numpy.nanmin(X, axis=0)

                # Set data type
                self.mean_ = self.mean_.astype(numpy.float64, copy=False)
                self.var_ = self.var_.astype(numpy.float64, copy=False)
                self.range_ = self.range_.astype(numpy.float64, copy=False)
            else:
                # Must be False for sparse
                self.mean_ = None  
                self.var_ = None
                weights = _check_sample_weight(sample_weight, X)
                sum_weights_nan = weights @ sparse_constructor(
                    (numpy.isnan(X.data), X.indices, X.indptr),
                    shape=X.shape)
                self.n_samples_seen_ += (
                    (numpy.sum(weights) - sum_weights_nan).astype(dtype)
                )
        else:
            # First pass
            if not hasattr(self, 'scale_'):
                self.mean_ = .0
                if self.with_range:
                    self.var_ = .0
                    self.range_ = .0
                else:
                    self.var_ = None
                    self.range_ = None

            if not self.with_mean and not self.with_range:
                self.mean_ = None
                self.var_ = None
                self.range_ = None
                self.n_samples_seen_ += X.shape[0] - numpy.isnan(X).sum(axis=0)

            else:
                self.mean_, self.var_, self.n_samples_seen_ = \
                    _incremental_mean_and_var(X, self.mean_, self.var_,
                                              self.n_samples_seen_)
                data_range = numpy.nanmax(X, axis=0)-numpy.nanmin(X, axis=0)
                self.range_ = _handle_zeros_in_scale(data_range, copy=True)
        # for backward-compatibility, reduce n_samples_seen_ to an integer
        # if the number of samples is the same for each feature (i.e. no
        # missing values)
        if numpy.ptp(self.n_samples_seen_) == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]

        if self.with_range:
            data_range = numpy.nanmax(X, axis=0)-numpy.nanmin(X, axis=0)
            self.scale_ = _handle_zeros_in_scale(data_range, copy=True)
        else:
            self.scale_ = None
        return self

    def transform(self, X, copy=None):
        """
        Transform data.

        Perform standardization by centering and scaling using the parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        copy : bool
            Copy the X matrix.
        
               
        Returns
        -------
        matrix_scaled : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Scaled version of the matrix.
        """
        # Check if fitted
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        # Validate data
        X = self._validate_data(X, reset=False,
                                accept_sparse='csr', copy=copy,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')
        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X = X - self.mean_
            if self.with_range:
                X = X / self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """
        Inverse transform data.

        Scale back the data to the original representation.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        copy : bool
            Copy the X matrix.
        
        Returns
        -------
        matrix_reverted : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Matrix with the scaling operation reverted.
        """

        # Check if fitted
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy, ensure_2d=False,
                        dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_range:
                X = X * self.scale_
            if self.with_mean:
                X = X + self.mean_
        return X

    def _more_tags(self):
        return {'allow_nan': True,
                'preserves_dtype': [numpy.float64, numpy.float32]}

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class CustomScalerPareto(TransformerMixin, BaseEstimator):
    """
    Pareto scaler.
    
    Custom scaler class for pipeline. Customized for Pareto scaling.
    
    Parameters
    ----------
    copy : bool
        Copy the array containing the data.
    with_mean : bool
        Perform mean centering.
    with_pareto : bool
        Scale the data.
    """
    def __init__(self, *, copy=True, with_mean=True, with_pareto=True):
        # Initialization
        self.with_mean = with_mean
        self.with_pareto = with_pareto
        self.copy = copy

    def _reset(self):
        # Checking attribute
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit transformer.
        
        Compute the mean and standard deviation from a dataset to use in future scaling operations.
                
        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        """
        Partial fit transformer.
        
        Performs online computation of mean and standard deviation on X for later scaling.
        All of X is processed as a single batch.
        This is intended for cases when `fit` is
        not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.

        Notes
        -----
        [1] Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms for computing the sample variance: Analysis and recommendations."The American Statistician 37.3 (1983): 242-247
        """
        first_call = not hasattr(self, "n_samples_seen_")
        # Validate data
        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan', reset=first_call)
        n_features = X.shape[1]

        # Check sample weights
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        # Set data type
        dtype = numpy.int64 if sample_weight is None else X.dtype
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = numpy.zeros(n_features, dtype=dtype)
        elif numpy.size(self.n_samples_seen_) == 1:
            self.n_samples_seen_ = numpy.repeat(
                self.n_samples_seen_, X.shape[1])
            self.n_samples_seen_ = \
                self.n_samples_seen_.astype(dtype, copy=False)

        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            sparse_constructor = (scipy.sparse.csr_matrix
                                  if X.format == 'csr' else scipy.sparse.csc_matrix)

            if self.with_pareto:
                # First pass
                if not hasattr(self, 'scale_'):
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        mean_variance_axis(X, axis=0, weights=sample_weight,
                                           return_sum_weights=True)
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_,
                                                weights=sample_weight)
                # Set data type
                self.mean_ = self.mean_.astype(numpy.float64, copy=False)
                self.var_ = self.var_.astype(numpy.float64, copy=False)
            else:
                # Must be False for sparse
                self.mean_ = None
                self.var_ = None
                weights = _check_sample_weight(sample_weight, X)
                sum_weights_nan = weights @ sparse_constructor(
                    (numpy.isnan(X.data), X.indices, X.indptr),
                    shape=X.shape)
                self.n_samples_seen_ += (
                    (numpy.sum(weights) - sum_weights_nan).astype(dtype)
                )
        else:
            # First pass
            if not hasattr(self, 'scale_'):
                self.mean_ = .0
                if self.with_pareto:
                    self.var_ = .0
                else:
                    self.var_ = None

            if not self.with_mean and not self.with_pareto:
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - numpy.isnan(X).sum(axis=0)

            else:
                self.mean_, self.var_, self.n_samples_seen_ = \
                    _incremental_mean_and_var(X, self.mean_, self.var_,
                                              self.n_samples_seen_)

        if numpy.ptp(self.n_samples_seen_) == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]

        if self.with_pareto:
            # Extract the list of near constant features on the raw variances,
            # before taking the square root.
            constant_mask = self.var_ < 10 * numpy.finfo(X.dtype).eps
            self.scale_ = _handle_zeros_in_scale(
                numpy.sqrt(numpy.sqrt(self.var_)), copy=False, constant_mask=constant_mask)
        else:
            self.scale_ = None

        return self

    def transform(self, X, copy=None):
        """
        Transform data.

        Perform standardization by centering and scaling using the parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        copy : bool
            Copy the X matrix.
        
               
        Returns
        -------
        matrix_scaled : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Scaled version of the matrix.
        """
        # Check if fitted
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        # Validate data
        X = self._validate_data(X, reset=False,
                                accept_sparse='csr', copy=copy,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')
        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X = X - self.mean_
            if self.with_pareto:
                X = X / self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """
        Inverse transform data.

        Scale back the data to the original representation.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        copy : bool
            Copy the X matrix.
        
        Returns
        -------
        matrix_reverted : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Matrix with the scaling operation reverted.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy, ensure_2d=False,
                        dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_pareto:
                X = X * self.scale_
            if self.with_mean:
                X = X + self.mean_
        return X

    def _more_tags(self):
        return {'allow_nan': True,
                'preserves_dtype': [numpy.float64, numpy.float32]}

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class CustomScalerVast(TransformerMixin, BaseEstimator):
    """
    Vast scaler.
    
    Custom scaler class for pipeline. Customized for Vast scaling.
    
    Parameters
    ----------
    copy : bool
        Copy the array containing the data.
    with_mean : bool
        Perform mean centering.
    with_vast : bool
        Scale the data.
    """
    def __init__(self, *, copy=True, with_mean=True, with_vast=True):
        # Initialization
        self.with_mean = with_mean
        self.with_vast = with_vast
        self.copy = copy

    def _reset(self):
        # Checking attribute, attributes set together
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit transformer.
        
        Compute the mean and standard deviation from a dataset to use in future scaling operations.
                
        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        """
        Partial fit transformer.
        
        Performs online computation of mean and standard deviation on X for later scaling.
        All of X is processed as a single batch.
        This is intended for cases when `fit` is
        not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.

        Notes
        -----
        [1] Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms for computing the sample variance: Analysis and recommendations."The American Statistician 37.3 (1983): 242-247
        """
        first_call = not hasattr(self, "n_samples_seen_")
        # Validate data
        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan', reset=first_call)
        n_features = X.shape[1]

        # Check sample weights
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        # Set data type
        dtype = numpy.int64 if sample_weight is None else X.dtype
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = numpy.zeros(n_features, dtype=dtype)
        elif numpy.size(self.n_samples_seen_) == 1:
            self.n_samples_seen_ = numpy.repeat(
                self.n_samples_seen_, X.shape[1])
            self.n_samples_seen_ = \
                self.n_samples_seen_.astype(dtype, copy=False)

        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            sparse_constructor = (scipy.sparse.csr_matrix
                                  if X.format == 'csr' else scipy.sparse.csc_matrix)

            if self.with_vast:
                # First pass
                if not hasattr(self, 'scale_'):
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        mean_variance_axis(X, axis=0, weights=sample_weight,
                                           return_sum_weights=True)
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_,
                                                weights=sample_weight)
                # Set data type
                self.mean_ = self.mean_.astype(numpy.float64, copy=False)
                self.var_ = self.var_.astype(numpy.float64, copy=False)
            else:
                # Must be False for sparse
                self.mean_ = None
                self.var_ = None
                weights = _check_sample_weight(sample_weight, X)
                sum_weights_nan = weights @ sparse_constructor(
                    (numpy.isnan(X.data), X.indices, X.indptr),
                    shape=X.shape)
                self.n_samples_seen_ += (
                    (numpy.sum(weights) - sum_weights_nan).astype(dtype)
                )
        else:
            # First pass
            if not hasattr(self, 'scale_'):
                self.mean_ = .0
                if self.with_vast:
                    self.var_ = .0
                else:
                    self.var_ = None

            if not self.with_mean and not self.with_vast:
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - numpy.isnan(X).sum(axis=0)

            else:
                self.mean_, self.var_, self.n_samples_seen_ = \
                    _incremental_mean_and_var(X, self.mean_, self.var_,
                                              self.n_samples_seen_)

        if numpy.ptp(self.n_samples_seen_) == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]

        if self.with_vast:
            self.scale_ = (1/numpy.sqrt(numpy.sqrt(self.var_)))*(self.mean_/numpy.sqrt(numpy.sqrt(self.var_)))
        else:
            self.scale_ = None
        return self

    def transform(self, X, copy=None):
        """
        Transform data.

        Perform standardization by centering and scaling using the parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        copy : bool
            Copy the X matrix.
        
               
        Returns
        -------
        matrix_scaled : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Scaled version of the matrix.
        """
        # Check if fitted
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        # Validate data
        X = self._validate_data(X, reset=False,
                                accept_sparse='csr', copy=copy,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')
        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X = X - self.mean_
            if self.with_vast:
                X = X / self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """
        Inverse transform data.

        Scale back the data to the original representation.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        copy : bool
            Copy the X matrix.
        
        Returns
        -------
        matrix_reverted : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Matrix with the scaling operation reverted.
        """
        # Check if fitted
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy, ensure_2d=False,
                        dtype=FLOAT_DTYPES, force_all_finite="allow-nan")
        
        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_vast:
                X = X * self.scale_
            if self.with_mean:
                X = X + self.mean_
        return X

    def _more_tags(self):
        return {'allow_nan': True,
                'preserves_dtype': [numpy.float64, numpy.float32]}

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class CustomScalerLevel(TransformerMixin, BaseEstimator):
    """
    Level scaler.
    
    Custom scaler class for pipeline. Customized for Level scaling.
    
    Parameters
    ----------
    copy : bool
        Copy the array containing the data.
    with_mean : bool
        Perform mean centering.
    with_level : bool
        Scale the data.
    """
    def __init__(self, *, copy=True, with_mean=True, with_level=True):
        # Initialization
        self.with_mean = with_mean
        self.with_level = with_level
        self.copy = copy

    def _reset(self):
        # Checking attribute, attributes set together
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit transformer.
        
        Compute the mean and standard deviation from a dataset to use in future scaling operations.
                
        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        """
        Partial fit transformer.
        
        Performs online computation of mean and standard deviation on X for later scaling.
        All of X is processed as a single batch.
        This is intended for cases when `fit` is
        not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.

        Notes
        -----
        [1] Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms for computing the sample variance: Analysis and recommendations."The American Statistician 37.3 (1983): 242-247
        """
        first_call = not hasattr(self, "n_samples_seen_")
        # Validate data
        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan', reset=first_call)
        n_features = X.shape[1]

        # Check sample weights
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        # Set data type
        dtype = numpy.int64 if sample_weight is None else X.dtype
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = numpy.zeros(n_features, dtype=dtype)
        elif numpy.size(self.n_samples_seen_) == 1:
            self.n_samples_seen_ = numpy.repeat(
                self.n_samples_seen_, X.shape[1])
            self.n_samples_seen_ = \
                self.n_samples_seen_.astype(dtype, copy=False)

        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            sparse_constructor = (scipy.sparse.csr_matrix
                                  if X.format == 'csr' else scipy.sparse.csc_matrix)

            if self.with_level:
                # First pass
                if not hasattr(self, 'scale_'):
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        mean_variance_axis(X, axis=0, weights=sample_weight,
                                           return_sum_weights=True)
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_,
                                                weights=sample_weight)
                # Set data type
                self.mean_ = self.mean_.astype(numpy.float64, copy=False)
                self.var_ = self.var_.astype(numpy.float64, copy=False)
            else:
                # Must be False for sparse
                self.mean_ = None
                self.var_ = None
                weights = _check_sample_weight(sample_weight, X)
                sum_weights_nan = weights @ sparse_constructor(
                    (numpy.isnan(X.data), X.indices, X.indptr),
                    shape=X.shape)
                self.n_samples_seen_ += (
                    (numpy.sum(weights) - sum_weights_nan).astype(dtype)
                )
        else:
            # First pass
            if not hasattr(self, 'scale_'):
                self.mean_ = .0
                if self.with_level:
                    self.var_ = .0
                else:
                    self.var_ = None

            if not self.with_mean and not self.with_level:
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - numpy.isnan(X).sum(axis=0)

            else:
                self.mean_, self.var_, self.n_samples_seen_ = \
                    _incremental_mean_and_var(X, self.mean_, self.var_,
                                              self.n_samples_seen_)

        if numpy.ptp(self.n_samples_seen_) == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]

        if self.with_level:
            self.scale_ = self.mean_.copy()
        else:
            self.scale_ = None

        return self

    def transform(self, X, copy=None):
        """
        Transform data.

        Perform standardization by centering and scaling using the parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        copy : bool
            Copy the X matrix.
        
               
        Returns
        -------
        matrix_scaled : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Scaled version of the matrix.
        """
        # Check if fitted
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        # Validate data
        X = self._validate_data(X, reset=False,
                                accept_sparse='csr', copy=copy,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')

        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X = X - self.mean_
            if self.with_level:
                X = X / self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """
        Inverse transform data.

        Scale back the data to the original representation.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        copy : bool
            Copy the X matrix.
        
        Returns
        -------
        matrix_reverted : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Matrix with the scaling operation reverted.
        """
        # Check if fitted
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy, ensure_2d=False,
                        dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        # Check sparsity
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_level:
                X = X * self.scale_
            if self.with_mean:
                X = X + self.mean_
        return X

    def _more_tags(self):
        return {'allow_nan': True,
                'preserves_dtype': [numpy.float64, numpy.float32]}

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class CustomScalerResponse(BaseEstimator, TransformerMixin):
    """
    Response scaler.
    
    Custom scaler class for pipeline. Customized for Response scaling.
    Necessary for binary and multiclass scaling.

    Parameters
    ----------
    copy : bool
        Copy the array containing the data.
    with_mean : bool
        Perform mean centering.
    with_std : bool
        Scale the data.
    """

    def __init__(self, scale_power=1, copy=True, with_mean=True, with_std=True):
        # Initialization
        self.scale_power = scale_power
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        # Checking attribute, attributes set together
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None):
        """
        Fit transformer.
        
        Compute the mean and standard deviation from a dataset to use in future scaling operations.
                
        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """
        Partial fit transformer.
        
        Performs online computation of mean and standard deviation on X for later scaling.
        All of X is processed as a single batch.
        This is intended for cases when `fit` is
        not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        
        Returns
        -------
        object_fitted : CustomScalerResponse
            Fitted transformer.

        Notes
        -----
        [1] Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms for computing the sample variance: Analysis and recommendations."The American Statistician 37.3 (1983): 242-247
        """

        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES)

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.with_std:
                # First pass
                if not hasattr(self, 'n_samples_seen_'):
                    self.mean_, self.var_ = mean_variance_axis(X, axis=0)
                    self.n_samples_seen_ = X.shape[0]
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_)
            else:
                self.mean_ = None
                self.var_ = None
        else:
            # First pass
            if not hasattr(self, 'n_samples_seen_'):
                self.mean_ = .0
                self.n_samples_seen_ = 0
                if self.with_std:
                    self.var_ = .0
                else:
                    self.var_ = None

            self.mean_, self.var_, self.n_samples_seen_ = \
                _incremental_mean_and_var(X, self.mean_, self.var_,
                                          self.n_samples_seen_)

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(numpy.sqrt(self.var_)) ** self.scale_power
        else:
            self.scale_ = None

        return self

    def transform(self, X, y=None, copy=None):
        """
        Transform data.

        Perform standardization by centering and scaling using the parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        y : None
            Passthrough for Scikit-learn ``Pipeline`` compatibility.
        copy : bool
            Copy the X matrix.
        
               
        Returns
        -------
        matrix_scaled : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Scaled version of the matrix.
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy

        X = check_array(X, accept_sparse='csr', copy=copy,
                        estimator=self, dtype=FLOAT_DTYPES)

        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X = X - self.mean_
            if self.with_std:
                X = X / self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """
        Inverse transform data.

        Scale back the data to the original representation.

        Parameters
        ----------
        X : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to scale.
        copy : bool
            Copy the X matrix.
        
        Returns
        -------
        matrix_reverted : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Matrix with the scaling operation reverted.
        """

        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        if scipy.sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if not scipy.sparse.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            X = numpy.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X = X * self.scale_
            if self.with_mean:
                X = X + self.mean_

        return X

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class ChemometricsPCA(BaseEstimator):
    """
    PCA object.
    
    ChemometricsPCA object - Wrapper for sklearn.decomposition PCA algorithms.

    Parameters
    ----------
    ncomps : int
        Number of PCA components.
    pca_algorithm : class
        scikit-learn PCA models (inheriting from _BasePCA).
    scaler : CustomScaler object
        Preprocessing objects or None.
    kwargs : pca_type_kwargs
        Keyword arguments to be passed during initialization of pca_algorithm.
    
    Raises
    ------
    TypeError
        If the pca_algorithm or scaler objects are not of the right class.
    """

    def __init__(self, ncomps=2, pca_algorithm=skPCA, scaler=CustomScalerAuto(), **pca_type_kwargs):
        try:
            # Perform the check with is instance but avoid abstract base class runs. PCA needs number of comps anyway!
            init_pca_algorithm = pca_algorithm(n_components=ncomps, **pca_type_kwargs)
            if not isinstance(init_pca_algorithm, (BaseEstimator, TransformerMixin)):
                raise TypeError("Scikit-learn model please")
            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = CustomScalerAuto(scale_power = 0, with_std=False)

            # TODO try adding partial fit methods
            # Add a check for partial fit methods? As in deploy partial fit child class if PCA is incremental??
            # By default it will work, but having the partial_fit function acessible might be usefull
            # Method hook in case the underlying pca algo allows partial fit?

            # The kwargs provided for the model are exactly the same as those
            # go and check for these examples the correct exception to throw when kwarg is not valid
            # TODO: Set the sklearn params for PCA to be a junction of the custom ones and the "core" params of model
            # overall aim is to make the object a "scikit-learn" object mimick
            self.pca_algorithm = init_pca_algorithm

            # Most initialized as None, before object is fitted.
            self.scores = None
            self.loadings = None
            self._ncomps = ncomps
            self._scaler = scaler
            self.cvParameters = None
            self.modelParameters = None
            self._isfitted = False
            self.x_raw = None
            self.y_raw = None
            self.y_raw_unique = None

        except TypeError as terp:
            print(terp.args[0])
            raise terp

    def fit(self, x, **fit_params):
        """
        Model fit.

        Perform model fitting on the provided X data matrix and calculate basic goodness-of-fit metrics.
        Equivalent to scikit-learn's default BaseEstimator method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to fit the PCA model.
        kwargs : fit_params
            Keyword arguments to be passed to the .fit() method of the core sklearn model.
        
        Raises
        ------
        ValueError
            If any problem occurs during fitting.
        """

        try:
            # This scaling check is always performed to ensure running model with scaling or with scaling == None
            # always give consistent results (same type of data scale expected for fitting,
            # returned by inverse_transform, etc
            if self.scaler is not None:
                xscaled = self.scaler.fit_transform(x)
                self.pca_algorithm.fit(xscaled, **fit_params)
                self.scores = self.pca_algorithm.transform(xscaled)
                ss = numpy.sum((xscaled - numpy.mean(xscaled, 0)) ** 2)
                predicted = self.pca_algorithm.inverse_transform(self.scores)
                rss = numpy.sum((xscaled - predicted) ** 2)
                # variance explained from scikit-learn stored as well
            else:
                self.pca_algorithm.fit(x, **fit_params)
                self.scores = self.pca_algorithm.transform(x)
                ss = numpy.sum((x - numpy.mean(x, 0)) ** 2)
                predicted = self.pca_algorithm.inverse_transform(self.scores)
                rss = numpy.sum((x - predicted) ** 2)
            self.modelParameters = {'R2X': 1 - (rss / ss), 'VarExp': self.pca_algorithm.explained_variance_,
                                    'VarExpRatio': self.pca_algorithm.explained_variance_ratio_}

            # For "Normalised" DmodX calculation
            resid_ssx = self._residual_ssx(x)
            s0 = numpy.sqrt(resid_ssx.sum()/((self.scores.shape[0] - self.ncomps - 1)*(x.shape[1] - self.ncomps)))
            self.modelParameters['S0'] = s0
            # Kernel PCA and other non-linear methods might not have explicit loadings - safeguard against this
            if hasattr(self.pca_algorithm, 'components_'):
                self.loadings = self.pca_algorithm.components_
            self._isfitted = True

        except ValueError as verr:
            raise verr

    def _partial_fit(self, x):
        """
        Under construction

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to fit the PCA model.
        """
        # TODO partial fit support
        return NotImplementedError

    def fit_transform(self, x, **fit_params):
        """
        Model fit and transform data.

        Fit a model and return the scores, as per the scikit-learn's TransformerMixin method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to fit the PCA model.
        kwargs : fit_params
            Keyword arguments to be passed to the .fit() method of the core sklearn model.
        
        Returns
        -------
        T : numpy.ndarray, shape [n_samples, n_comps]
            PCA projections (scores) corresponding to the samples in X.

        Raises
        ------
        ValueError
            If there are problems with the input or during model fitting.
        """
        try:
            self.fit(x, **fit_params)
            return self.transform(x)
        except ValueError as exp:
            raise exp

    def transform(self, x):
        """
        Transform data.

        Calculate the projections (scores) of the x data matrix. Similar to scikit-learn's TransformerMixin method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to transform.
        kwargs : transform_params
            Optional keyword arguments to be passed to the transform method.
        
        Returns
        -------
        scores : numpy.ndarray, shape [n_samples, n_comps]
            PCA projections (scores) corresponding to the samples in X.

        Raises
        ------
        ValueError
            If there are problems with the input or during model fitting.
        """
        try:
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
                return self.pca_algorithm.transform(xscaled)
            else:
                return self.pca_algorithm.transform(x)
        except ValueError as verr:
            raise verr

    def inverse_transform(self, scores):
        """
        Inverse transform data.

        Transform scores to the original data space using the principal component loadings.
        Similar to scikit-learn's default TransformerMixin method.

        Parameters
        ----------
        scores : numpy.ndarray, shape [n_samples, n_comps]
            The projections (scores) to be converted back to the original data space.
        
        Returns
        -------
        matrix : numpy.ndarray, shape [n_samples, n_features]
            Data matrix in the original data space.

        Raises
        ------
        ValueError
            If the dimensions of score mismatch the number of components in the model.
        """
        # Scaling check for consistency
        if self.scaler is not None:
            xinv_prescaled = self.pca_algorithm.inverse_transform(scores)
            xinv = self.scaler.inverse_transform(xinv_prescaled)
            return xinv
        else:
            return self.pca_algorithm.inverse_transform(scores)

    def score(self, x, sample_weight=None):
        """
        Score model.

        Return the average log-likelihood of all samples. 
        Same as the underlying score method from the scikit-learn PCA objects.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to score model on.
        sample_weight : numpy.ndarray
            Optional sample weights during scoring.
        
        Returns
        -------
        score : float
            Average log-likelihood over all samples.

        Raises
        ------
        ValueError
            If there are problems with the input.
        """
        try:
            # Not all sklearn pca objects have a "score" method...
            score_method = getattr(self.pca_algorithm, "score", None)
            if not callable(score_method):
                raise NotImplementedError
            # Scaling check for consistency
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
                return self.pca_algorithm.score(xscaled, sample_weight)
            else:
                return self.pca_algorithm.score(x, sample_weight)
        except ValueError as verr:
            raise verr

    def _press_impute_pinv(self, x, var_to_pred):
        """
        Single value imputation.

        Single value imputation method, essential to use in the cross-validation.
        In theory can also be used to do missing data imputation.
        Based on the Eigenvector_PRESS calculation.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix in the original data space.
        var_to_pred : int
            Which variable is to be imputed from the others.
        
        Returns
        -------
        x_imputed : numpy.ndarray, shape [n_samples, n_features]
            Data matrix in the original data space.

        Raises
        ------
        ValueError
            If there is any error during the imputation process.
        
        Notes
        -----
        [1] Bro et al, Cross-validation of component models: A critical look at current methods, Analytical and Bioanalytical Chemistry 2008
        [2] Amoeba's answer on CrossValidated: http://stats.stackexchange.com/a/115477
        """

        # TODO Double check improved algorithms and methods for PRESS estimation for PCA in general
        # TODO Implement Camacho et al, column - erfk to increase computational efficiency
        # TODO check bi-cross validation
        try:
            # Scaling check for consistency
            if self.scaler is not None:
                xscaled = self.scaler.transform(x)
            else:
                xscaled = x
            # Following from reference 1
            to_pred = numpy.delete(xscaled, var_to_pred, axis=1)
            topred_loads = numpy.delete(self.loadings.T, var_to_pred, axis=0)
            imputed_x = numpy.dot(numpy.dot(to_pred, numpy.linalg.pinv(topred_loads).T), self.loadings)
            if self.scaler is not None:
                imputed_x = self.scaler.inverse_transform(imputed_x)
            return imputed_x
        except ValueError as verr:
            raise verr

    @property
    def ncomps(self):
        try:
            return self._ncomps
        except AttributeError as atre:
            raise atre

    @ncomps.setter
    def ncomps(self, ncomps):
        """
        Setter for number of components.

        Parameters
        ----------
        ncomps : int
            Number of PCA components to use in the model.

        Raises
        ------
        AttributeError
            If there is a problem changing the number of components and resetting the model.
        """
        # To ensure changing number of components effectively resets the model
        try:
            self._ncomps = ncomps
            self.pca_algorithm = clone(self.pca_algorithm, safe=True)
            self.pca_algorithm.n_components = ncomps
            self.modelParameters = None
            self.loadings = None
            self.scores = None
            self.cvParameters = None
            self.x_raw = None
            self.y_raw = None
            self.y_raw_unique = None
            return None
        except AttributeError as atre:
            raise atre

    @property
    def scaler(self):
        try:
            return self._scaler
        except AttributeError as atre:
            raise atre

    @scaler.setter
    def scaler(self, scaler):
        """
        Setter for the model scaler.

        Parameters
        ----------
        scaler : CustomScaler object
            Scaling/preprocessing object or None.

        Raises
        ------
        AttributeError
            If there is a problem changing the scaler and resetting the model.
        TypeError
            If the new scaler provided is not a valid object.
        """
        try:
            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = CustomScalerAuto(scale_power = 0, with_std=False)

            self._scaler = scaler
            self.pca_algorithm = clone(self.pca_algorithm, safe=True)
            self.modelParameters = None
            self.loadings = None
            self.scores = None
            self.cvParameters = None
            self.x_raw = None
            self.y_raw = None
            self.y_raw_unique = None
            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    def hotelling_T2(self, comps=None, alpha=0.05):
        """
        Hotelling T2 ellipse.

        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.

        Parameters
        ----------
        comps : list
            List of components in 2D.
        alpha : float
            Significance level.
        
        Returns
        -------
        radii : numpy.ndarray
            The Hotelling T2 ellipsoid radii at vertex.

        Raises
        ------
        AtributeError
            If the model is not fitted.
        ValueError
            If the components requested are higher than the number of components in the model.
        TypeError
            If comps is not None or list/numpy 1d array and alpha a float.
        """
        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")
            nsamples = self.scores.shape[0]
            if comps is None:
                ncomps = self.ncomps
                ellips = self.scores[:, range(self.ncomps)] ** 2
                ellips = 1 / nsamples * (ellips.sum(0))
            else:
                ncomps = len(comps)
                ellips = self.scores[:, comps] ** 2
                ellips = 1 / nsamples * (ellips.sum(0))

            # F stat
            fs = (nsamples - 1) / nsamples * ncomps * (nsamples ** 2 - 1) / (nsamples * (nsamples - ncomps))
            fs = fs * scipy.stats.f.ppf(1-alpha, ncomps, nsamples - ncomps)

            hoteling_t2 = list()
            for comp in range(ncomps):
                hoteling_t2.append(numpy.sqrt((fs * ellips[comp])))

            return numpy.array(hoteling_t2)

        except AttributeError as atre:
            raise atre
        except ValueError as valerr:
            raise valerr
        except TypeError as typerr:
            raise typerr

    def dmodx(self, x):
        """
        Normalised DmodX measure.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.

        Returns
        -------
        dmodx : numpy.ndarray
            The Normalised DmodX measure for each sample.
        """
        resids_ssx = self._residual_ssx(x)
        s = numpy.sqrt(resids_ssx/(self.loadings.shape[1] - self.ncomps))
        dmodx = numpy.sqrt((s/self.modelParameters['S0'])**2)
        return dmodx

    def leverages(self):
        """
        Calculate the leverages for each observation.
        
        Returns
        -------
        H : numpy.ndarray
            The leverage (H) for each observation.
        """
        return numpy.diag(numpy.dot(self.scores, numpy.dot(numpy.linalg.inv(numpy.dot(self.scores.T, self.scores)), self.scores.T)))

    def outlier(self, x, comps=None, measure='T2', alpha=0.05):
        """
        Check outlier.

        Use the Hotelling T2 or DmodX measure and F statistic to screen for outlier candidates.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        comps : list
            List of components in 2D.
        measure : str
            Hotelling T2 (T2) or DmodX
        alpha : float
            Significance level
        
        Returns
        -------
        index_row : list
            List with row indices of X matrix.
        """
        try:
            if measure == 'T2':
                scores = self.transform(x)
                t2 = self.hotelling_T2(comps=comps)
                outlier_idx = numpy.where(((scores ** 2) / t2 ** 2).sum(axis=1) > 1)[0]
            elif measure == 'DmodX':
                dmodx = self.dmodx(x)
                dcrit = self._dmodx_fcrit(x, alpha)
                outlier_idx = numpy.where(dmodx > dcrit)[0]
            else:
                print("Select T2 (Hotelling T2) or DmodX as outlier exclusion criteria")
            return outlier_idx
        except Exception as exp:
            raise exp

    def cross_validation(self, x, y, cv_method=model_selection.KFold(7, shuffle=True), outputdist=False, press_impute=False):
        """
        Cross validation.

        Cross-validation method for the model. Calculates cross-validated estimates for Q2X and other
        model parameters using row-wise cross validation.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        cv_method : BaseCrossValidator
            An instance of a scikit-learn CrossValidator object.
        outputdist : bool
            Output the whole distribution for the cross validated parameters.
        press_impute : bool
            Use imputation of test set observations instead of row wise cross-validation.

        Returns
        -------
        cv_params : dict
            Adds a dictionary cvParameters to the object, containing the cross validation results.

        Raises
        ------
        TypeError
            If the cv_method passed is not a scikit-learn CrossValidator object.
        ValueError
            If the x data matrix is invalid.
        """
        try:
            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False or self.loadings is None:
                self.fit(x)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            cv_pipeline = deepcopy(self)

            # Initialise predictive residual sum of squares variable (for whole CV routine)
            total_press = 0
            # Calculate Sum of Squares SS in whole dataset
            ss = numpy.sum((cv_pipeline.scaler.transform(x)) ** 2)
            # Initialise list for loadings and for the VarianceExplained in the test set values
            # Check if model has loadings, as in case of kernelPCA these are not available
            if hasattr(self.pca_algorithm, 'components_'):
                loadings = []

            # cv_varexplained_training is a list containing lists with the SingularValue/Variance Explained metric
            # as obtained in the training set during fitting.
            # cv_varexplained_test is a single R2X measure obtained from using the
            # model fitted with the training set in the test set.
            cv_varexplained_training = []
            cv_varexplained_test = []
            cv_train_scores = []
            cv_test_scores = []
            # Default version (press_impute = False) will perform
            #  Row/Observation-Wise CV - Faster computationally, but has some limitations
            # See Bro R. et al, Cross-validation of component models: A critical look at current methods,
            # Analytical and Bioanalytical Chemistry 2008
            # press_impute method requires computational optimization, and is under construction
            for xtrain, xtest in cv_method.split(x, y):
                cv_pipeline.fit(x[xtrain, :])
                # Calculate R2/Variance Explained in test set
                # To calculate an R2X in the test set

                xtest_scaled = cv_pipeline.scaler.transform(x[xtest, :])

                tss = numpy.sum((xtest_scaled) ** 2)
                # Append the var explained in training set for this round and loadings for this round
                cv_varexplained_training.append(cv_pipeline.pca_algorithm.explained_variance_ratio_)
                cv_train_scores.append([(item[0], item[1]) for item in zip(xtrain, cv_pipeline.scores)])
                
                pred_scores = cv_pipeline.transform(x[xtest, :])
                cv_test_scores.append([(item[0], item[1]) for item in zip(xtest, pred_scores)])

                if hasattr(self.pca_algorithm, 'components_'):
                    loadings.append(cv_pipeline.loadings)

                if press_impute is True:
                    press_testset = 0
                    for column in range(0, x[xtest, :].shape[1]):
                        xpred = cv_pipeline.scaler.transform(cv_pipeline._press_impute_pinv(x[xtest, :], column))
                        press_testset += numpy.sum(numpy.square(xtest_scaled[:, column] - xpred[:, column]))
                    cv_varexplained_test.append(1 - (press_testset / tss))
                    total_press += press_testset
                else:
                    # RSS for row wise cross-validation
                    pred_scores = cv_pipeline.transform(x[xtest, :])
                    pred_x = cv_pipeline.scaler.transform(cv_pipeline.inverse_transform(pred_scores))
                    rss = numpy.sum(numpy.square(xtest_scaled - pred_x))
                    total_press += rss
                    cv_varexplained_test.append(1 - (rss / tss))

            #print(cv_scores_training)
            # Create matrices for each component loading containing the cv values in each round
            # nrows = nrounds, ncolumns = n_variables
            # Check that the PCA model has loadings
            if hasattr(self.pca_algorithm, 'components_'):
                cv_loads = []
                for comp in range(0, self.ncomps):
                    cv_loads.append(numpy.array([x[comp] for x in loadings]))

                # Align loadings due to sign indeterminacy.
                # The solution followed here is to select the sign that gives a more similar profile to the
                # Loadings calculated with the whole data.
                # TODO add scores for CV scores, but still need to check the best way to do it properly
                # Don't want to enforce the common "just average everything" and interpret score plot behaviour...
                for cvround in range(0, cv_method.get_n_splits()):
                    for currload in range(0, self.ncomps):
                        choice = numpy.argmin(numpy.array([numpy.sum(numpy.abs(self.loadings - cv_loads[currload][cvround, :])),
                                                     numpy.sum(
                                                         numpy.abs(self.loadings - cv_loads[currload][cvround, :] * -1))]))
                        if choice == 1:
                            cv_loads[currload][cvround, :] = -1 * cv_loads[currload][cvround, :]

                            list_train_score = []
                            for idx, array in cv_train_scores[cvround]:
                                array[currload] = array[currload]*-1
                                list_train_score.append((idx,array))

                            list_test_score = []
                            for idx, array in cv_test_scores[cvround]:
                                array[currload] = array[currload]*-1
                                list_test_score.append((idx,array))

                            cv_train_scores[cvround] =  list_train_score
                            cv_test_scores[cvround] =  list_test_score
                        else:
                            list_train_score = []
                            for idx, array in cv_train_scores[cvround]:
                                array[currload] = array[currload]
                                list_train_score.append((idx,array))

                            list_test_score = []
                            for idx, array in cv_test_scores[cvround]:
                                array[currload] = array[currload]
                                list_test_score.append((idx,array))

                            cv_train_scores[cvround] =  list_train_score
                            cv_test_scores[cvround] =  list_test_score

            # Calculate total sum of squares
            # Q^2X
            q_squared = 1 - (total_press / ss)
            # Assemble the dictionary and data matrices
            if self.cvParameters is not None:
                self.cvParameters['Mean_VarExpRatio_Training'] = numpy.array(cv_varexplained_training).mean(axis=0)
                self.cvParameters['Stdev_VarExpRatio_Training'] = numpy.array(cv_varexplained_training).std(axis=0)
                self.cvParameters['Mean_VarExp_Test'] = numpy.mean(cv_varexplained_test)
                self.cvParameters['Stdev_VarExp_Test'] = numpy.std(cv_varexplained_test)
                self.cvParameters['Q2X'] = q_squared
            else:
                self.cvParameters = {'Mean_VarExpRatio_Training': numpy.array(cv_varexplained_training).mean(axis=0),
                                     'Stdev_VarExpRatio_Training': numpy.array(cv_varexplained_training).std(axis=0),
                                     'Mean_VarExp_Test': numpy.mean(cv_varexplained_test),
                                     'Stdev_VarExp_Test': numpy.std(cv_varexplained_test),
                                      'Q2X': q_squared}
            if outputdist is True:
                self.cvParameters['CV_VarExpRatio_Training'] = cv_varexplained_training
                self.cvParameters['CV_VarExp_Test'] = cv_varexplained_test
                self.cvParameters['CV_TrainScores'] = cv_train_scores
                self.cvParameters['CV_TestScores'] = cv_test_scores
                
            # Check that the PCA model has loadings
            if hasattr(self.pca_algorithm, 'components_'):
                self.cvParameters['Mean_Loadings'] = [numpy.mean(x, 0) for x in cv_loads]
                self.cvParameters['Stdev_Loadings'] = [numpy.std(x, 0) for x in cv_loads]
                if outputdist is True:
                    self.cvParameters['CV_Loadings'] = cv_loads
            return None

        except TypeError as terp:
            raise terp
        except ValueError as verr:
            raise verr

    def _residual_ssx(self, x):
        """
        Residual Sum of Squares.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        
        Returns
        -------
        RSSX : numpy.ndarray
            The residual Sum of Squares per sample.
        """
        pred_scores = self.transform(x)

        x_reconstructed = self.scaler.transform(self.inverse_transform(pred_scores))
        xscaled = self.scaler.transform(x)
        residuals = numpy.sum((xscaled - x_reconstructed)**2, axis=1)
        return residuals

    def x_residuals(self, x, scale=True):
        """
        Residual Sum of Squares.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        scale : bool
            Return the residuals in the scale the model is using or in the raw data scale.
        
        Returns
        -------
        x_residuals : numpy.ndarray
            X matrix model residuals.
        """
        pred_scores = self.transform(x)
        x_reconstructed = self.scaler.transform(self.inverse_transform(pred_scores))
        xscaled = self.scaler.transform(x)

        x_residuals = numpy.sum((xscaled - x_reconstructed)**2, axis=1)
        if scale:
            x_residuals = self.scaler.inverse_transform(x_residuals)

        return x_residuals

    #@staticmethod
    #def stop_cond(model, x):
    #    stop_check = getattr(model, modelParameters)
    #    if stop_check > 0:
    #        return True
    #    else:
    #        return False

    def _screecv_optimize_ncomps(self, x, total_comps=5, cv_method=model_selection.KFold(7, shuffle=True), stopping_condition=None):
        """
        Optimize number of components.

        Routine to optimize number of components quickly using Cross validation and stabilization of Q2X.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        total_comps : int
            Maximal number of components.
        cv_method : BaseCrossValidator
            An instance of a scikit-learn CrossValidator object.
        stopping_condition : float or None
            Stopping condition.

        Returns
        -------
        results_dict : dict
            Adds scree parameters to cvParameters dictionary.
        """
        models = list()

        for ncomps in range(1, total_comps + 1):

            currmodel = deepcopy(self)
            currmodel.ncomps = ncomps
            currmodel.fit(x)
            currmodel.cross_validation(x, outputdist=False, cv_method=cv_method, press_impute=False)
            models.append(currmodel)

            # Stopping condition on Q2, assuming stopping_condition is a float encoding percentage of increase from
            # previous Q2X
            # Exclude first component since there is nothing to compare with...
            if isinstance(stopping_condition, float) and ncomps > 1:
                previous_q2 = models[ncomps - 2].cvParameters['Q2X']
                current_q2 = models[ncomps - 1].cvParameters['Q2X']

                if (current_q2 - previous_q2)/abs(previous_q2) < stopping_condition:
                    # Stop the loop
                    models.pop()
                    break
            # Flexible case to be implemented, to allow many other stopping conditions
            elif callable(stopping_condition):
                pass

        q2 = numpy.array([x.cvParameters['Q2X'] for x in models])
        r2 = numpy.array([x.modelParameters['R2X'] for x in models])

        results_dict = {'R2X_Scree': r2, 'Q2X_Scree': q2, 'Scree_n_components': len(r2)}
        # If cross-validation has been called
        if self.cvParameters is not None:
            self.cvParameters['R2X_Scree'] = r2
            self.cvParameters['Q2X_Scree'] = q2
            self.cvParameters['Scree_n_components'] = len(r2)
        # In case cross_validation wasn't called before.
        else:
            self.cvParameters = {'R2X_Scree': r2, 'Q2X_Scree': q2, 'Scree_n_components': len(r2)}

        return results_dict

    def _dmodx_fcrit(self, x, alpha=0.05):
        """
        Critical DmodX value.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        alpha : float
            Significance level.
        
        Returns
        -------
        dmodx_crit : float
            Critical DmodX value.

        Notes
        -----
        [1] Faber, Nicolaas (Klaas) M., Degrees of freedom for the residuals of a principal component analysis - A clarification, Chemometrics and Intelligent Laboratory Systems 2008
        """
        dmodx_fcrit = scipy.stats.f.ppf(1 - alpha, x.shape[1] - self.ncomps - 1,
                         (x.shape[0] - self.ncomps - 1) * (x.shape[1] - self.ncomps))

        return dmodx_fcrit

    def permutationtest_loadings(self, x, nperms=1000):
        """
        Permutation test loadings.

        Permutation test to assess significance of magnitude of value for variable in component loading vector.
        Can be used to test importance of variable to the loading vector.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        nperms : int
            Number of permutations.
                
        Returns
        -------
        H0 : numpy.ndarray, shape [ncomps, n_perms, n_features]
            Permuted null distribution for loading vector values.

        Raises
        ------
        ValueError
            If there is a problem with the input x data or during the procedure.
        """
        # TODO: Work in progress, more as a curiosity
        try:
            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False or self.loadings is None:
                self.fit(x)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            permute_class = deepcopy(self)
            # Initalise list for loading distribution
            perm_loads = [numpy.zeros((nperms, x.shape[1]))] * permute_class.ncomps
            for permutation in range(0, nperms):
                for var in range(0, x.shape[1]):
                    # Copy original column order, shuffle array in place...
                    orig = numpy.copy(x[:, var])
                    numpy.random.shuffle(x[:, var])
                    # ... Fit model and replace original data
                    permute_class.fit(x)
                    x[:, var] = orig
                    # Store the loadings for each permutation component-wise
                    for loading in range(0, permute_class.ncomps):
                        perm_loads[loading][permutation, var] = permute_class.loadings[loading][var]

            # Align loadings due to sign indeterminacy.
            # Solution provided is to select the sign that gives a more similar profile to the
            # Loadings calculated with the whole data.
            for perm_n in range(0, nperms):
                for currload in range(0, permute_class.ncomps):
                    choice = numpy.argmin(numpy.array([numpy.sum(numpy.abs(self.loadings - perm_loads[currload][perm_n, :])),
                                                 numpy.sum(numpy.abs(
                                                     self.loadings - perm_loads[currload][perm_n, :] * -1))]))
                    if choice == 1:
                        perm_loads[currload][perm_n, :] = -1 * perm_loads[currload][perm_n, :]
            return perm_loads
        except ValueError as verr:
            raise verr

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class ChemometricsPLS(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    PLS object.
    
    ChemometricsPLS object - Wrapper for sklearn.cross_decomposition PLS algorithms.
    This object is designed to fit flexibly both PLSRegression with one or multiple Y and PLSCanonical, both
    with either NIPALS or SVD. PLS-SVD doesn't calculate the same type of model parameters, and should
    not be used with this object.
    For PLSRegression/PLS1/PLS2 and PLSCanonical/PLS-C2A/PLS-W2A, the actual components
    found may differ (depending on type of deflation, etc), and this has to be taken into consideration,
    but the actual nomenclature/definitions should be the "same".
    Nomenclature is as follows:
    X - T Scores - Projections of X, called T
    Y - U Scores - Projections of Y, called U
    X - Loadings P - Vector/multivariate directions associated with T on X are called P (equivalent to PCA)
    Y - Loadings Q - Vector/multivariate directions associated with U on Y are called q
    X - Weights W - Weights/directions of maximum covariance with Y of the X block are called W
    Y - Weights C - Weights/directions of maximum covariance with X of the Y block block are called C
    X - Rotations W*/Ws/R - The rotation of X variables to LV space pinv(WP')W
    Y - Rotations C*/Cs - The rotation of Y variables to LV space pinv(CQ')C
    T = X W(P'W)^-1 = XW* (W* : p x k matrix)
    U = Y C(Q'C)^-1 = YC* (C* : q x k matrix)
    Loadings and weights after the first component do not represent
    the original variables. The SIMPLS-style (similar interpretation but not the same Rotations that would be obtained from 
    using the SIMPLS algorithm) W*/Ws and C*/Cs act as weight vectors
    which relate to the original X and Y variables, and not to their deflated versions.
    For more information see Sijmen de Jong, "SIMPLS: an alternative approach to partial least squares regression", Chemometrics
    and Intelligent Laboratory Systems 1992
    "Inner" relation regression coefficients of T b_t: U = Tb_t
    "Inner" relation regression coefficients of U b_U: T = Ub_u
    These are obtained by regressing the U's and T's, applying standard linear regression to them.
    B = pinv(X'X)X'Y
    b_t = pinv(T'T)T'U
    b_u = pinv(U'U)U'T
    or in a form usually seen in PLS NIPALS algorithms: b_t are the betas from regressing T on U - t'u/u'u
    and b_u are the betas from regressing U on T - u't/t't
    In summary, there are various ways to approach the model. Following a general nomenclature applicable
    for both single and block Y:
    For predictions, the model assumes the Latent variable formulation and uses an "inner relation"
    between the latent variable projections, where U = Tb_t and T = Ub_u.
    Therefore, prediction using the so-called "mixed relations" (relate T with U and subsequently Y/relate
    U with T and subsequently X), works through the following formulas
    Y = T*b_t*C' + G
    X = U*b_u*W' + H
    The b_u and b_s are effectively "regression coefficients" between the latent variable scores
    
    In parallel, we can think in terms of "outer relations", data decompositions or linear approximations to
    the original data blocks, similar to PCA components
    Y = UQ' + F
    X = TP' + E
    For PLS regression with single y, Y = UC' + F = Y = UQ' + F, due to Q = C, but not necessarily true for
    multi Y, so Q' is used here. Notice that this formula cannot be used directly to
    predict Y from X and vice-versa, the inner relation regression using latent variable scores is necessary.
    
    Finally, assuming PLSRegression (single or multi Y, but asymmetric deflation):
    The PLS model can be approached from a multivariate regression/regularized regression point of view,
    where Y is related to the original X variables, through regression coefficients Beta,
    bypassing the latent variable definition and concepts.
    Y = XBQ', Y = XB, where B are the regression coefficients and B = W*Q' (the W*/ws is the SIMPLS-like R rotation,
    the x_rotation in sklearn default PLS algorithms).
    The Betas (regression coefficients) obtained in this manner directly relate the original X variables
    to the prediction of Y.
    
    This MLR (multivariate linear regression) approach to PLS has the advantage of exposing the PLS betas and PLS mechanism
    as a biased regression applying a degree of shrinkage, which decreases with the number of components
    all the way up to B(OLS), when Number of Components = number of variables/columns.
    
    Parameters
    ----------
    ncomps : int
        Number of PLS components.
    pls_algorithm : class
        scikit-learn PLS algorithm to use - PLSRegression or PLSCanonical are supported.
    xscaler : CustomScaler object
        Scaler object for X data matrix.
    yscaler : CustomScaler object
        Scaler object for y data matrix.

    kwargs : pls_type_kwargs
        Keyword arguments to be passed during initialization of pls_algorithm.
    
    Raises
    ------
    TypeError
        If the pls_algorithm or scaler objects are not of the right class.
    
    Notes
    -----
    [1] Frank, Ildiko E. Friedman, Jerome H., A Statistical View of Some Chemometrics Regression Tools, 1993
    [2] de Jong, PLS shrinks, Journal of Chemometrics, 1995 
    [3] Nicole Kramer, An Overview on the Shrinkage Properties of Partial Least Squares Regression, Computational Statistics, 2007
    """

    def __init__(self, ncomps=2, pls_algorithm=PLSRegression, xscaler=CustomScalerAuto(), yscaler=None,
                 **pls_type_kwargs):
        try:
            # Perform the check with is instance but avoid abstract base class runs.
            pls_algorithm = pls_algorithm(ncomps, scale=False, **pls_type_kwargs)
            if not isinstance(pls_algorithm, (BaseEstimator)):
                raise TypeError("Scikit-learn model please")
            if not (isinstance(xscaler, TransformerMixin) or xscaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if not (isinstance(yscaler, TransformerMixin) or yscaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            # 2 blocks of data = two scaling options
            if xscaler is None:
                xscaler = CustomScalerAuto(scale_power = 0, with_std=False)
                # Force scaling to false, as this will be handled by the provided scaler or not
            if yscaler is None:
                yscaler = CustomScalerResponse(scale_power = 0, with_std=False)

            self.pls_algorithm = pls_algorithm
            self.x_raw = None
            self.y_raw = None
            # Most initialized as None, before object is fitted...
            self.scores_t = None
            self.scores_u = None
            self.weights_w = None
            self.weights_c = None
            self.loadings_p = None
            self.loadings_q = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.b_u = None
            self.b_t = None
            self.beta_coeffs = None

            self._ncomps = ncomps
            self._x_scaler = xscaler
            self._y_scaler = yscaler
            self.cvParameters = None
            self.modelParameters = None
            self.permutationParameters = None
            self.bootstrapParameters = None
            self._isfitted = False

        except TypeError as terp:
            print(terp.args[0])

    def fit(self, x, y, **fit_params):
        """
        Model fit.

        Perform model fitting on the provided X data matrix and calculate basic goodness-of-fit metrics.
        Similar to scikit-learn's default BaseEstimator method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to fit the PLS model.
        y : numpy.ndarray, shape [n_samples, n_responses]
            Response matrix to fit the PLS model.
        kwargs : fit_params
            Keyword arguments to be passed to the .fit() method of the core sklearn model.
        
        Raises
        ------
        ValueError
            If any problem occurs during fitting.
        """
        try:
            # This scaling check is always performed to ensure running model with scaling or with scaling == None
            # always gives consistent results (the same type of data scale used fitting will be expected or returned
            # by all methods of the ChemometricsPLS object)
            # For no scaling, mean centering is performed nevertheless - sklearn objects
            # do this by default, this is solely to make everything ultra clear and to expose the
            # interface for potential future modification
            # Comply with the sklearn-scaler behaviour convention
            # Save data
            self.x_raw = x
            self.y_raw = y
            self.y_raw_unique = [item for item in pandas.Series(y).unique()]

            if y.ndim == 1:
                y = y.reshape(-1, 1)
            # Not so important as don't expect a user applying a single x variable to a multivariate regression
            # method, but for consistency/testing purposes
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            xscaled = self.x_scaler.fit_transform(x)
            yscaled = self.y_scaler.fit_transform(y)

            self.pls_algorithm.fit(xscaled, yscaled, **fit_params)

            # Expose the model parameters
            self.loadings_p = self.pls_algorithm.x_loadings_
            self.loadings_q = self.pls_algorithm.y_loadings_
            self.weights_w = self.pls_algorithm.x_weights_
            self.weights_c = self.pls_algorithm.y_weights_
            self.rotations_ws = self.pls_algorithm.x_rotations_
            # scikit learn sets the rotation, causing a discrepancy between the scores calculated during fitting and the transform method
            # for now, we calculate the rotation and override it: C* = pinv(CQ')C
            self.rotations_cs = numpy.dot(numpy.linalg.pinv(numpy.dot(self.weights_c, self.loadings_q.T)), self.weights_c)
            self.scores_t = self.pls_algorithm.x_scores_
            self.scores_u = self.pls_algorithm.y_scores_
            self.b_u = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(self.scores_u.T, self.scores_u)), self.scores_u.T),
                              self.scores_t)
            self.b_t = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(self.scores_t.T, self.scores_t)), self.scores_t.T),
                              self.scores_u)
            self.beta_coeffs = self.pls_algorithm.coef_
            # Needs to come here for the method shortcuts down the line to work...
            self._isfitted = True

            # Calculate RSSy/RSSx, R2Y/R2X
            R2Y = ChemometricsPLS.score(self, x=x, y=y, block_to_score='y')
            R2X = ChemometricsPLS.score(self, x=x, y=y, block_to_score='x')

            # Obtain residual sum of squares for whole data set and per component
            cm_fit = self._cummulativefit(x, y)

            self.modelParameters = {'R2Y': R2Y, 'R2X': R2X, 'SSX': cm_fit['SSX'], 'SSY': cm_fit['SSY'],
                                    'SSXcomp': cm_fit['SSXcomp'], 'SSYcomp': cm_fit['SSYcomp']}

            resid_ssx = self._residual_ssx(x)
            s0 = numpy.sqrt(resid_ssx.sum() / ((self.scores_t.shape[0] - self.ncomps - 1) * (x.shape[1] - self.ncomps)))
            self.modelParameters['S0X'] = s0

        except ValueError as verr:
            raise verr

    def fit_transform(self, x, y, **fit_params):
        """
        Model fit and transform data.

        Fit a model and return the scores, as per the scikit-learn's TransformerMixin method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to fit the PLS model.
        y : numpy.ndarray, shape [n_samples, n_responses]
            Response matrix to fit the PLS model.
        kwargs : fit_params
            Keyword arguments to be passed to the .fit() method of the core sklearn model.
        
        Returns
        -------
        (T,U) : tuple of numpy.ndarray, shape [[n_tscores], [n_uscores]]
            Latent Variable scores (T) for the X matrix and for the Y vector/matrix (U).

        Raises
        ------
        ValueError
            If there are problems with the input or during model fitting.
        """

        try:
            self.fit(x, y, **fit_params)
            return self.transform(x, y=None), self.transform(x=None, y=y)

        except ValueError as verr:
            raise verr

    def transform(self, x=None, y=None):
        """
        Transform data.

        Calculate the scores for a data block from the original data. Equivalent to sklearn's TransformerMixin method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to fit the PLS model.
        y : numpy.ndarray, shape [n_samples, n_responses]
            Response matrix to fit the PLS model.
        
        Returns
        -------
        (T,U) : tuple of numpy.ndarray, shape [[n_tscores], [n_uscores]]
            Latent Variable scores (T) for the X matrix and for the Y vector/matrix (U).

        Raises
        ------
        ValueError
            If dimensions of input data are mismatched.
        AttributeError
            When calling the method before the model is fitted.
        """
        try:
            # Check if model is fitted
            if self._isfitted is True:
                # If X and Y are passed, complain and do nothing
                if (x is not None) and (y is not None):
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif (x is None) and (y is None):
                    raise ValueError('yy')
                # If Y is given, return U
                elif x is None:
                    if y.ndim == 1:
                        y = y.reshape(-1, 1)

                    yscaled = self.y_scaler.transform(y)
                    # Taking advantage of rotations_y
                    # Otherwise this would be the full calculation U = Y*pinv(CQ')*C
                    U = numpy.dot(yscaled, self.rotations_cs)
                    return U

                # If X is given, return T
                elif y is None:
                    # Not so important as don't expect a user applying a single x variable to a multivariate regression
                    # method, but for consistency/testing purposes
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)

                    xscaled = self.x_scaler.transform(x)
                    # Taking advantage of already calculated rotation_x
                    # Otherwise this would be would the full calculation T = X*pinv(WP')*W
                    T = numpy.dot(xscaled, self.rotations_ws)
                    return T
            else:
                raise AttributeError('Model not fitted')

        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    def inverse_transform(self, t=None, u=None):
        """
        Inverse transform data.

        Transform scores to the original data space using their corresponding loadings.
        Similar to scikit-learn's default TransformerMixin method.

        Parameters
        ----------
        t : numpy.ndarray, shape [n_samples, n_comps] or None
            T scores corresponding to the X data matrix.
        u : numpy.ndarray, shape [n_samples, n_responses] or None
            U scores corresponding to the y data matrix.
        
        Returns
        -------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.

        Raises
        ------
        ValueError
            If dimensions of input data are mismatched.
        """
        try:
            if self._isfitted is True:
                if t is not None and u is not None:
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif t is None and u is None:
                    raise ValueError('yy')
                # If T is given, return U
                elif t is not None:
                    # Calculate X from T using X = TP'
                    xpred = numpy.dot(t, self.loadings_p.T)
                    if self.x_scaler is not None:
                        xscaled = self.x_scaler.inverse_transform(xpred)
                    else:
                        xscaled = xpred

                    return xscaled
                # If U is given, return T
                elif u is not None:
                    # Calculate Y from U - using Y = UQ'
                    ypred = numpy.dot(u, self.loadings_q.T)
                    if self.y_scaler is not None:
                        yscaled = self.y_scaler.inverse_transform(ypred)
                    else:
                        yscaled = ypred

                    return yscaled

        except ValueError as verr:
            raise verr

    def score(self, x, y, block_to_score='y', sample_weight=None):
        """
        Score model.

        Predict and calculate the R2 for the model using one of the data blocks (X or Y) provided.
        Equivalent to the scikit-learn RegressorMixin score method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.
        block_to_score : str
            Which of the data blocks (X or Y) to calculate the R2 goodness of fit.
        sample_weight : numpy.ndarray, shape [n_samples] or None
            Optional sample weights to use in scoring.
        
        Returns
        -------
        R2Y : float
            The model's R2Y, calculated by predicting Y from X and scoring.
        R2X : float
            The model's R2X, calculated by predicting X from Y and scoring.

        Raises
        ------
        ValueError
            If block to score argument is not acceptable or date mismatch issues with the provided data.
        """
        # TODO: Check how to implement sample_weight, which is expected in scikit-learn methods, for PLS algorithms
        try:
            if block_to_score not in ['x', 'y']:
                raise ValueError("x or y are the only accepted values for block_to_score")
            # Comply with the sklearn scaler behaviour
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            # Not so important as don't expect a user applying a single x variable to a multivariate regression
            # method, but for consistency/testing purposes
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            # Calculate RSSy/RSSx, R2Y/R2X
            if block_to_score == 'y':
                yscaled = deepcopy(self.y_scaler).fit_transform(y)
                # Calculate total sum of squares of X and Y for R2X and R2Y calculation
                tssy = numpy.sum(numpy.square(yscaled))
                ypred = self.y_scaler.transform(ChemometricsPLS.predict(self, x, y=None))
                rssy = numpy.sum(numpy.square(yscaled - ypred))
                R2Y = 1 - (rssy / tssy)
                return R2Y
            # The prediction here of both X and Y is done using the other block of data only
            # so these R2s can be interpreted as as a "classic" R2, and not as a proportion of variance modelled
            # Here we use X = Ub_uW', as opposed to (X = TP').
            else:
                xscaled = deepcopy(self.x_scaler).fit_transform(x)
                # Calculate total sum of squares of X and Y for R2X and R2Y calculation
                xpred = self.x_scaler.transform(ChemometricsPLS.predict(self, x=None, y=y))
                tssx = numpy.sum(numpy.square(xscaled))
                rssx = numpy.sum(numpy.square(xscaled - xpred))
                R2X = 1 - (rssx / tssx)
                return R2X
        except ValueError as verr:
            raise verr

    def predict(self, x=None, y=None):
        """
        Predict scores.

        Predict the values in one data block using the other. Same as its scikit-learn's RegressorMixin namesake method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.
       
        Returns
        -------
        predicted : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Predicted data block (X or Y) obtained from the other data block.

        Raises
        ------
        ValueError
            If no data matrix is passed, or dimensions mismatch issues with the provided data.
        AttributeError
            Calling the method without fitting the model before.
        """
        try:
            if self._isfitted is True:
                if (x is not None) and (y is not None):
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif (x is None) and (y is None):
                    raise ValueError('yy')
                # Predict Y from X
                elif x is not None:
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    xscaled = self.x_scaler.transform(x)

                    # Using Betas to predict Y directly
                    predicted = numpy.dot(xscaled, self.beta_coeffs)
                    if predicted.ndim == 1:
                        predicted = predicted.reshape(-1, 1)
                    predicted = self.y_scaler.inverse_transform(predicted)
                    return predicted
                # Predict X from Y
                elif y is not None:
                    # Going through calculation of U and then X = Ub_uW'
                    u_scores = ChemometricsPLS.transform(self, x=None, y=y)
                    predicted = numpy.dot(numpy.dot(u_scores, self.b_u), self.weights_w.T)
                    if predicted.ndim == 1:
                        predicted = predicted.reshape(-1, 1)
                    predicted = self.x_scaler.inverse_transform(predicted)
                    return predicted
            else:
                raise AttributeError("Model is not fitted")
        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    @property
    def ncomps(self):
        try:
            return self._ncomps
        except AttributeError as atre:
            raise atre

    @ncomps.setter
    def ncomps(self, ncomps=1):
        """
        Setter for number of components.

        Parameters
        ----------
        ncomps : int
            Number of PLS components to use in the model.

        Raises
        ------
        AttributeError
            If there is a problem changing the number of components and resetting the model.
        """
        # To ensure changing number of components effectively resets the model
        try:
            self._ncomps = ncomps
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.pls_algorithm.n_components = ncomps
            self.x_raw = None
            self.y_raw = None
            self.y_raw_unique = None
            self.loadings_p = None
            self.scores_t = None
            self.scores_u = None
            self.loadings_q = None
            self.weights_c = None
            self.weights_w = None
            self.rotations_cs = None
            self.rotations_ws = None
            self.cvParameters = None
            self.modelParameters = None
            self.permutationParameters = None
            self.bootstrapParameters = None
            self.b_t = None
            self.b_u = None
            self.beta_coeffs = None

            return None
        except AttributeError as atre:
            raise atre

    @property
    def x_scaler(self):
        try:
            return self._x_scaler
        except AttributeError as atre:
            raise atre

    @x_scaler.setter
    def x_scaler(self, scaler):
        """
        Setter for the X data block scaler.

        Parameters
        ----------
        scaler : CustomScaler object
            Scaling/preprocessing object or None.

        Raises
        ------
        AttributeError
            If there is a problem changing the scaler and resetting the model.
        TypeError
            If the new scaler provided is not a valid object.
        """
        try:

            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = CustomScalerAuto(scale_power = 0, with_std=False)

            self._x_scaler = scaler
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.x_raw = None
            self.y_raw = None
            self.y_raw_unique = None
            self.cvParameters = None
            self.modelParameters = None
            self.permutationParameters = None
            self.bootstrapParameters = None
            self.loadings_p = None
            self.weights_w = None
            self.weights_c = None
            self.loadings_q = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.scores_t = None
            self.scores_u = None
            self.b_t = None
            self.b_u = None
            self.beta_coeffs = None

            return None
        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    @property
    def y_scaler(self):
        try:
            return self._y_scaler
        except AttributeError as atre:
            raise atre

    @y_scaler.setter
    def y_scaler(self, scaler):
        """
        Setter for the Y data block scaler.

        Parameters
        ----------
        scaler : CustomScaler object
            Scaling/preprocessing object or None.

        Raises
        ------
        AttributeError
            If there is a problem changing the scaler and resetting the model.
        TypeError
            If the new scaler provided is not a valid object.
        """
        try:
            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = CustomScalerResponse(scale_power = 0, with_std=False)

            self._y_scaler = scaler
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.x_raw = None
            self.y_raw = None
            self.y_raw_unique = None
            self.cvParameters = None
            self.modelParameters = None
            self.permutationParameters = None
            self.bootstrapParameters = None
            self.loadings_p = None
            self.weights_w = None
            self.weights_c = None
            self.loadings_q = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.scores_t = None
            self.scores_u = None
            self.b_t = None
            self.b_u = None
            self.beta_coeffs = None

            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    def VIP(self):
        """
        Variable Importance on Projection.

        Output the Variable importance for projection metric (VIP). 
        With the default values it is calculated using the x variable weights and the variance explained of y. 
        Default mode is recommended (mode = 'w' and direction = 'y').
        
        Parameters
        ----------
        mode : str
            The type of model parameter to use in calculating the VIP. 
            Default value is weights (w), and other acceptable arguments are p, ws, cs, c and q.
        direction : str
            The data block to be used to calculated the model fit and regression sum of squares.
        
        Returns
        -------
        VIP : numpy.ndarray, shape [n_features]
            The vector with the calculated VIP values.

        Raises
        ------
        ValueError
            If mode or direction is not a valid option.
        AttributeError
            Calling method without a fitted model.
        """
        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")

            y = self.y_raw
            y_unique = self.y_raw_unique
            # Get data
            t = self.scores_t
            w = self.weights_w
            q = self.loadings_q
            p, h = w.shape
            # Preallocate array
            vips = numpy.zeros((p,q.shape[0]))
            # Multi-class VIP
            if self.n_classes > 2:
                # Cycle classes
                for k in y_unique:
                    vips_k = numpy.zeros((p,))
                    # SSY and SSYcum is different
                    ind_k = numpy.where(y==k)[0]
                    t_k = t[ind_k,:]
                    q_k = q
                    # Calculate SSY and SSYcum
                    SSY = numpy.diag(t_k.T @ t_k @ q_k.T @ q_k).reshape(h, -1)
                    SSYcum = numpy.sum(SSY)
                    for i in range(p):
                        weight = numpy.array([(w[i,j] / numpy.linalg.norm(w[:,j]))**2 for j in range(h)])
                        vips_k[i] = numpy.sqrt(p*(SSY.T @ weight)/SSYcum)
                    vips[:,k] = vips_k
            else:
                # Calculate SSY and SSYcum
                SSY = numpy.diag(t.T @ t @ q.T @ q).reshape(h, -1)
                SSYcum = numpy.sum(SSY)

                # Full model
                vips = numpy.zeros((p,))
                for i in range(p):
                    weight = numpy.array([(w[i,j] / numpy.linalg.norm(w[:,j]))**2 for j in range(h)])
                    vips[i] = numpy.sqrt(p*(SSY.T @ weight)/SSYcum)
            return vips
        except AttributeError as atter:
            raise atter
        except ValueError as verr:
            raise verr

    def hotelling_T2(self, comps=None, alpha=0.05):
        """
        Hotelling T2 ellipse.

        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.

        Parameters
        ----------
        comps : list
            List of components in 2D.
        alpha : float
            Significance level.
        
        Returns
        -------
        radii : numpy.ndarray
            The Hotelling T2 ellipsoid radii at vertex.

        Raises
        ------
        AtributeError
            If the model is not fitted.
        ValueError
            If the components requested are higher than the number of components in the model.
        TypeError
            If comps is not None or list/numpy 1d array and alpha a float.
        """
        
        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")

            nsamples = self.scores_t.shape[0]

            if comps is None:
                ncomps = self.ncomps
                ellips = self.scores_t[:, range(self.ncomps)] ** 2
            else:
                ncomps = len(comps)
                ellips = self.scores_t[:, comps] ** 2

            ellips = 1 / nsamples * (ellips.sum(0))

            # F stat
            a = (nsamples - 1) / nsamples * ncomps * (nsamples ** 2 - 1) / (nsamples * (nsamples - ncomps))
            a = a * st.f.ppf(1-alpha, ncomps, nsamples - ncomps)

            hoteling_t2 = list()
            for comp in range(ncomps):
                hoteling_t2.append(numpy.sqrt((a * ellips[comp])))

            return numpy.array(hoteling_t2)

        except AttributeError as atre:
            raise atre
        except ValueError as valerr:
            raise valerr
        except TypeError as typerr:
            raise typerr

    def dmodx(self, x):
        """
        Normalised DmodX measure.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        scale : bool
            Return the residuals in the scale the model is using or in the raw data scale.
        
        Returns
        -------
        dmodx : numpy.ndarray
            The Normalised DmodX measure for each sample.
        """
        resids_ssx = self._residual_ssx(x)
        s = numpy.sqrt(resids_ssx/(self.loadings_p.shape[0] - self.ncomps))
        dmodx = numpy.sqrt((s/self.modelParameters['S0X'])**2)
        return dmodx

    def leverages(self, block='X'):
        """
        Calculate the leverages for each observation.
        
        Returns
        -------
        H : numpy.ndarray
            The leverage (H) for each observation.
        """
        # TODO check with matlab and simca
        try:
            if block == 'X':
                return numpy.dot(self.scores_t, numpy.dot(numpy.linalg.inv(numpy.dot(self.scores_t.T, self.scores_t), self.scores_t.T)))
            elif block == 'Y':
                return numpy.dot(self.scores_u, numpy.dot(numpy.linalg.inv(numpy.dot(self.scores_u.T, self.scores_u), self.scores_u.T)))
            else:
                raise ValueError
        except ValueError as verr:
            raise ValueError('block option must be either X or Y')

    def outlier(self, x, comps=None, measure='T2', alpha=0.05):
        """
        Check outlier.

        Use the Hotelling T2 or DmodX measure and F statistic to screen for outlier candidates.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        comps : list
            List of components in 2D.
        measure : str
            Hotelling T2 (T2) or DmodX
        alpha : float
            Significance level
        
        Returns
        -------
        index_row : list
            List with row indices of X matrix.
        """
        try:
            if measure == 'T2':
                scores = self.transform(x)
                t2 = self.hotelling_T2(comps=comps)
                outlier_idx = numpy.where(((scores ** 2) / t2 ** 2).sum(axis=1) > 1)[0]
            elif measure == 'DmodX':
                dmodx = self.dmodx(x)
                dcrit = st.f.ppf(1 - alpha, x.shape[1] - self.ncomps,
                                 (x.shape[0] - self.ncomps - 1) * (x.shape[1] - self.ncomps))
                outlier_idx = numpy.where(dmodx > dcrit)[0]
            else:
                print("Select T2 (Hotelling T2) or DmodX as outlier exclusion criteria")
            return outlier_idx
        except Exception as exp:
            raise exp

    def cross_validation(self, x, y, cv_method=model_selection.KFold(7, shuffle=True), outputdist=False,
                         **crossval_kwargs):
        """
        Cross validation.

        Cross-validation method for the model. Calculates Q2 and cross-validated estimates for all model parameters.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.
        cv_method : BaseCrossValidator or BaseShuffleSplit
            An instance of a scikit-learn CrossValidator object.
        outputdist : bool
            Output the whole distribution for the cross validated parameters.
        crossval_kwargs : kwargs
            Keyword arguments to be passed to the sklearn.Pipeline during cross-validation.

        Returns
        -------
        cv_params : dict
            Adds a dictionary cvParameters to the object, containing the cross validation results.

        Raises
        ------
        TypeError
            If the cv_method passed is not a scikit-learn CrossValidator object.
        ValueError
            If the x and y data matrix is invalid.
        """
        try:
            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False:
                self.fit(x, y)

            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            cv_pipeline = deepcopy(self)
            ncvrounds = cv_method.get_n_splits()

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1

            if y.ndim > 1:
                y_nvars = y.shape[1]
            else:
                y_nvars = 1
                y = y.reshape(-1, 1)

            # Initialize list structures to contain the fit
            cv_loadings_p = numpy.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_loadings_q = numpy.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_weights_w = numpy.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_weights_c = numpy.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_rotations_ws = numpy.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_rotations_cs = numpy.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_betacoefs = numpy.zeros((ncvrounds, y_nvars, x_nvars))
            cv_vipsw = numpy.zeros((ncvrounds, y_nvars, x_nvars))

            cv_train_scores_t = []
            cv_train_scores_u = []
            cv_test_scores_t = []
            cv_test_scores_u = []

            # Initialise predictive residual sum of squares variable (for whole CV routine)
            pressy = 0
            pressx = 0

            # Calculate Sum of Squares SS in whole dataset for future calculations
            ssx = numpy.sum(numpy.square(cv_pipeline.x_scaler.fit_transform(x)))
            ssy = numpy.sum(numpy.square(cv_pipeline.y_scaler.fit_transform(y)))

            # As assessed in the test set..., opposed to PRESS
            R2X_training = numpy.zeros(ncvrounds)
            R2Y_training = numpy.zeros(ncvrounds)
            # R2X and R2Y assessed in the test set
            R2X_test = numpy.zeros(ncvrounds)
            R2Y_test = numpy.zeros(ncvrounds)

            for cvround, train_testidx in enumerate(cv_method.split(x, y)):
                # split the data explicitly
                train = train_testidx[0]
                test = train_testidx[1]

                # Check dimensions for the indexing
                if y_nvars == 1:
                    ytrain = y[train]
                    ytest = y[test]
                else:
                    ytrain = y[train, :]
                    ytest = y[test, :]
                if x_nvars == 1:
                    xtrain = x[train]
                    xtest = x[test]
                else:
                    xtrain = x[train, :]
                    xtest = x[test, :]

                cv_pipeline.fit(xtrain, ytrain, **crossval_kwargs)
                # Prepare the scaled X and Y test data
                # If testset_scale is True, these are scaled individually...

                # Comply with the sklearn scaler behaviour
                if ytest.ndim == 1:
                    ytest = ytest.reshape(-1, 1)
                    ytrain = ytrain.reshape(-1, 1)
                if xtest.ndim == 1:
                    xtest = xtest.reshape(-1, 1)
                    xtrain = xtrain.reshape(-1, 1)
                # Fit the training data

                xtest_scaled = cv_pipeline.x_scaler.transform(xtest)
                ytest_scaled = cv_pipeline.y_scaler.transform(ytest)

                R2X_training[cvround] = cv_pipeline.score(xtrain, ytrain, 'x')
                R2Y_training[cvround] = cv_pipeline.score(xtrain, ytrain, 'y')

                ypred = cv_pipeline.predict(x=xtest, y=None)
                xpred = cv_pipeline.predict(x=None, y=ytest)

                xpred = cv_pipeline.x_scaler.transform(xpred).squeeze()

                ypred = cv_pipeline.y_scaler.transform(ypred).squeeze()
                ytest_scaled = ytest_scaled.squeeze()

                curr_pressx = numpy.sum(numpy.square(xtest_scaled - xpred))
                curr_pressy = numpy.sum(numpy.square(ytest_scaled - ypred))

                R2X_test[cvround] = cv_pipeline.score(xtest, ytest, 'x')
                R2Y_test[cvround] = cv_pipeline.score(xtest, ytest, 'y')

                pressx += curr_pressx
                pressy += curr_pressy

                ###
                cv_train_scores_t.append([(item[0], item[1]) for item in zip(train, cv_pipeline.scores_t)])
                cv_train_scores_u.append([(item[0], item[1]) for item in zip(train, cv_pipeline.scores_u)])

                pred_scores_t = cv_pipeline.transform(x = xtest, y = None)
                pred_scores_u = cv_pipeline.transform(x = None, y = ytest)

                cv_test_scores_t.append([(item[0], item[1]) for item in zip(test, pred_scores_t)])
                cv_test_scores_u.append([(item[0], item[1]) for item in zip(test, pred_scores_u)])
                ###

                cv_loadings_p[cvround, :, :] = cv_pipeline.loadings_p
                cv_loadings_q[cvround, :, :] = cv_pipeline.loadings_q
                cv_weights_w[cvround, :, :] = cv_pipeline.weights_w
                cv_weights_c[cvround, :, :] = cv_pipeline.weights_c
                cv_rotations_ws[cvround, :, :] = cv_pipeline.rotations_ws
                cv_rotations_cs[cvround, :, :] = cv_pipeline.rotations_cs
                cv_betacoefs[cvround, :, :] = cv_pipeline.beta_coeffs.T
                cv_vipsw[cvround, :, :] = cv_pipeline.VIP().T


            # TODO CV scores done properly
            # Align model parameters to account for sign indeterminacy.
            # The criteria here used is to select the sign that gives a more similar profile (by L1 distance) to the loadings fitted
            # on the model fitted with the whole data. Any other parameter can be used, but since the loadings in X capture
            # the covariance structure in X data block, in theory they should have more pronounced features even in cases of
            # null X-Y association, making the sign flip more resilient.
            for cvround in range(0, ncvrounds):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = numpy.argmin(
                        numpy.array([numpy.sum(numpy.abs(self.loadings_p[:, currload] - cv_loadings_p[cvround, :, currload])),
                                  numpy.sum(numpy.abs(
                                      self.loadings_p[:, currload] - cv_loadings_p[cvround, :, currload] * -1))]))
                    if choice == 1:
                        cv_loadings_p[cvround, :, currload] = -1 * cv_loadings_p[cvround, :, currload]
                        cv_loadings_q[cvround, :, currload] = -1 * cv_loadings_q[cvround, :, currload]
                        cv_weights_w[cvround, :, currload] = -1 * cv_weights_w[cvround, :, currload]
                        cv_weights_c[cvround, :, currload] = -1 * cv_weights_c[cvround, :, currload]
                        cv_rotations_ws[cvround, :, currload] = -1 * cv_rotations_ws[cvround, :, currload]
                        cv_rotations_cs[cvround, :, currload] = -1 * cv_rotations_cs[cvround, :, currload]
                        #cv_train_scores_t.append([*zip(train, -1 * cv_pipeline.scores_t)])
                        #cv_train_scores_u.append([*zip(train, -1 * cv_pipeline.scores_u)])
                        #cv_test_scores_t.append([*zip(test, -1 * cv_pipeline.scores_t)])
                        #cv_test_scores_u.append([*zip(test, -1 * cv_pipeline.scores_u)])
                    else:
                        None
                        #cv_train_scores_t.append([*zip(train, cv_pipeline.scores_t)])
                        #cv_train_scores_u.append([*zip(train, cv_pipeline.scores_u)])
                        #cv_test_scores_t.append([*zip(test, cv_pipeline.scores_t)])
                        #cv_test_scores_u.append([*zip(test, cv_pipeline.scores_u)])

            # Calculate total sum of squares
            q_squaredy = 1 - (pressy / ssy)
            q_squaredx = 1 - (pressx / ssx)

            # Store everything...
            self.cvParameters = {
                'Q2X': q_squaredx, 'Q2Y': q_squaredy, 
                'MeanR2X_Training': numpy.mean(R2X_training), 'StdevR2X_Training': numpy.std(R2X_training),
                'MeanR2Y_Training': numpy.mean(R2Y_training), 'StdevR2Y_Training': numpy.std(R2Y_training),
                'MeanR2X_Test': numpy.mean(R2X_test), 'StdevR2X_Test': numpy.std(R2X_test),
                'MeanR2Y_Test': numpy.mean(R2Y_test), 'StdevR2Y_Test': numpy.std(R2Y_test),
                'Mean_Loadings_q': cv_loadings_q.mean(0), 'Stdev_Loadings_q': cv_loadings_q.std(0), 
                'Mean_Loadings_p': cv_loadings_p.mean(0), 'Stdev_Loadings_p': cv_loadings_q.std(0), 
                'Mean_Weights_c': cv_weights_c.mean(0), 'Stdev_Weights_c': cv_weights_c.std(0), 
                'Mean_Weights_w': cv_weights_w.mean(0), 'Stdev_Weights_w': cv_weights_w.std(0), 
                'Mean_Rotations_ws': cv_rotations_ws.mean(0), 'Stdev_Rotations_ws': cv_rotations_ws.std(0),
                'Mean_Rotations_cs': cv_rotations_cs.mean(0), 'Stdev_Rotations_cs': cv_rotations_cs.std(0), 
                'Mean_Beta': cv_betacoefs.mean(0), 'Stdev_Beta': cv_betacoefs.std(0), 
                'Mean_VIP': cv_vipsw.mean(0), 'Stdev_VIP': cv_vipsw.std(0)
                }
            # TODO Investigate a better way to average this properly
            # Projection to a global "model"?
            # Means and standard deviations...
            # self.cvParameters['Mean_Scores_t'] = cv_scores_t.mean(0)
            # self.cvParameters['Stdev_Scores_t'] = cv_scores_t.std(0)
            # self.cvParameters['Mean_Scores_u'] = cv_scores_u.mean(0)
            # self.cvParameters['Stdev_Scores_u'] = cv_scores_u.std(0)
            # Save everything found during CV
            if outputdist is True:
                self.cvParameters['CVR2X_Training'] = R2X_training
                self.cvParameters['CVR2Y_Training'] = R2Y_training
                self.cvParameters['CVR2X_Test'] = R2X_test
                self.cvParameters['CVR2Y_Test'] = R2Y_test
                self.cvParameters['CV_Loadings_q'] = cv_loadings_q
                self.cvParameters['CV_Loadings_p'] = cv_loadings_p
                self.cvParameters['CV_Weights_c'] = cv_weights_c
                self.cvParameters['CV_Weights_w'] = cv_weights_w
                self.cvParameters['CV_Rotations_ws'] = cv_rotations_ws
                self.cvParameters['CV_Rotations_cs'] = cv_rotations_cs
                self.cvParameters['CV_Train_Scores_t'] = cv_train_scores_t
                self.cvParameters['CV_Train_Scores_u'] = cv_test_scores_u
                self.cvParameters['CV_Beta'] = cv_betacoefs
                self.cvParameters['CV_VIPw'] = cv_vipsw

            return None

        except TypeError as terp:
            raise terp

    def _residual_ssx(self, x):
        """
        Residual Sum of Squares.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        
        Returns
        -------
        RSSX : numpy.ndarray
            The residual Sum of Squares per sample.
        """
        pred_scores = self.transform(x)

        x_reconstructed = self.x_scaler.transform(self.inverse_transform(pred_scores))
        xscaled = self.x_scaler.transform(x)
        residuals = numpy.sum(numpy.square(xscaled - x_reconstructed), axis=1)
        return residuals

    def permutation_test(self, x, y, nperms=1000, cv_method=model_selection.KFold(7, shuffle=True), outputdist=False, **permtest_kwargs):
        """
        Permutation test.

        Permutation test for the classifier and most model parameters.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.
        nperms : int
            Number of permutations.
        cv_method : BaseCrossValidator or BaseShuffleSplit
            An instance of a scikit-learn CrossValidator object.
        outputdist : bool
            Output the permutation test parameters.
        permtest_kwargs : kwargs
            Keyword arguments to be passed to the sklearn.Pipeline during cross-validation.

        Returns
        -------
        perm_params : dict
            Adds a dictionary permParameters to the object, containing the permutation test results.

        Raises
        ------
        ValueError
            If the x and y data matrix is invalid.
        """
        try:
            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False or self.loadings_p is None:
                self.fit(x, y, **permtest_kwargs)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            permute_class = deepcopy(self)

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1

            if y.ndim > 1:
                y_nvars = y.shape[1]
            else:
                y_nvars = 1

            # Initialize data structures for permuted distributions
            perm_loadings_p = numpy.zeros((nperms, x_nvars, self.ncomps))
            perm_loadings_q = numpy.zeros((nperms, y_nvars, self.ncomps))
            perm_weights_w = numpy.zeros((nperms, x_nvars, self.ncomps))
            perm_weights_c = numpy.zeros((nperms, y_nvars, self.ncomps))
            perm_rotations_ws = numpy.zeros((nperms, x_nvars, self.ncomps))
            perm_rotations_cs = numpy.zeros((nperms, y_nvars, self.ncomps))
            perm_beta = numpy.zeros((nperms, y_nvars, x_nvars))
            perm_vipsw = numpy.zeros((nperms, y_nvars, x_nvars))

            perm_R2Y = numpy.zeros(nperms)
            perm_R2X = numpy.zeros(nperms)
            perm_Q2Y = numpy.zeros(nperms)
            perm_Q2X = numpy.zeros(nperms)

            for permutation in range(0, nperms):
                # Copy original column order, shuffle array in place...
                perm_y = numpy.random.permutation(y)
                # ... Fit model and replace original data
                permute_class.fit(x, perm_y, **permtest_kwargs)
                permute_class.cross_validation(x, perm_y, cv_method=cv_method, **permtest_kwargs)
                perm_R2Y[permutation] = permute_class.modelParameters['R2Y']
                perm_R2X[permutation] = permute_class.modelParameters['R2X']
                perm_Q2Y[permutation] = permute_class.cvParameters['Q2Y']
                perm_Q2X[permutation] = permute_class.cvParameters['Q2X']

                # Store the loadings for each permutation component-wise
                perm_loadings_q[permutation, :, :] = permute_class.loadings_q
                perm_loadings_p[permutation, :, :] = permute_class.loadings_p
                perm_weights_c[permutation, :, :] = permute_class.weights_c
                perm_weights_w[permutation, :, :] = permute_class.weights_w
                perm_rotations_cs[permutation, :, :] = permute_class.rotations_cs
                perm_rotations_ws[permutation, :, :] = permute_class.rotations_ws
                perm_beta[permutation, :, :] = permute_class.beta_coeffs.T
                perm_vipsw[permutation, :, :] = permute_class.VIP().T

            # Align model parameters due to sign indeterminacy.
            # Solution provided is to select the sign that gives a more similar profile to the
            # Loadings calculated with the whole data.
            for perm_round in range(0, nperms):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = numpy.argmin(numpy.array(
                        [numpy.sum(numpy.abs(self.loadings_p[:, currload] - perm_loadings_p[perm_round, :, currload])),
                         numpy.sum(numpy.abs(self.loadings_p[:, currload] - perm_loadings_p[perm_round, :, currload] * -1))]))
                    if choice == 1:
                        perm_loadings_p[perm_round, :, currload] = -1 * perm_loadings_p[perm_round, :, currload]
                        perm_loadings_q[perm_round, :, currload] = -1 * perm_loadings_q[perm_round, :, currload]
                        perm_weights_w[perm_round, :, currload] = -1 * perm_weights_w[perm_round, :, currload]
                        perm_weights_c[perm_round, :, currload] = -1 * perm_weights_c[perm_round, :, currload]
                        perm_rotations_ws[perm_round, :, currload] = -1 * perm_rotations_ws[perm_round, :, currload]
                        perm_rotations_cs[perm_round, :, currload] = -1 * perm_rotations_cs[perm_round, :, currload]

            # Pack everything into a nice data structure and return
            # Calculate p-value for Q2Y as well
            permutationTest = dict()
            permutationTest['R2Y'] = perm_R2Y
            permutationTest['R2X'] = perm_R2X
            permutationTest['Q2Y'] = perm_Q2Y
            permutationTest['Q2X'] = perm_Q2X
            permutationTest['R2Y_Test'] = perm_R2Y_test
            permutationTest['R2X_Test'] = perm_R2X_test
            permutationTest['Loadings_p'] = perm_loadings_p
            permutationTest['Loadings_q'] = perm_loadings_q
            permutationTest['Weights_c'] = perm_weights_c
            permutationTest['Weights_w'] = perm_weights_w
            permutationTest['Rotations_ws'] = perm_rotations_ws
            permutationTest['Rotations_cs'] = perm_rotations_cs
            permutationTest['Beta'] = perm_beta
            permutationTest['VIPw'] = perm_vipsw

            #obs_q2y = self.cvParameters['Q2Y']
            #pvals = dict()
            #pvals['Q2Y'] = (len(numpy.where(perm_Q2Y >= obs_q2y)) + 1) / (nperms + 1)
            #obs_r2y = self.cvParameters['R2Y_Test']
            #pvals['R2Y_Test'] = (len(numpy.where(perm_R2Y_test >= obs_r2y)) + 1) / (nperms + 1)
            return None#permutationTest, pvals

        except ValueError as exp:
            raise exp

    def bootstrap_test(self, x, y, nboots=100, stratify = True, outputdist=False, **bootstrap_kwargs):
        """
        Bootstrapping.

        Bootstrapping for confidence intervalls of the classifier and most model parameters.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.
        nboots : int
            Number of bootstraps.
        stratify : bool
            Use stratification.
        outputdist : bool
            Output the bootstrapping parameters.
        bootstrap_kwargs : kwargs
            Keyword arguments to be passed to the sklearn.Pipeline during cross-validation.

        Returns
        -------
        perm_params : dict
            Adds a dictionary permParameters to the object, containing the permutation test results.

        Raises
        ------
        ValueError
            If the x and y data matrix is invalid.
        """
        try:
            # Check if global model is fitted... and if not, fit it to provided x and y
            if self._isfitted is False or self.loadings_p is None:
                self.fit(x, y, **bootstrap_kwargs)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            bootstrap_pipeline = deepcopy(self)
            n_samples = int(0.7*len(y))
            seed = numpy.random.seed(None)
            bootidx = []
            bootidx_oob = []
            for i in range(nboots):
                #bootidx_i = numpy.random.choice(len(self.Y), len(self.Y))
                if stratify == True:
                    bootidx_i = sklearn.utils.resample(list(range(len(y))), n_samples = n_samples, stratify=y)
                else:
                    bootidx_i = sklearn.utils.resample(list(range(len(y))), n_samples = n_samples)
                bootidx_oob_i = numpy.array(list(set(range(len(y))) - set(bootidx_i)))
                bootidx.append(bootidx_i)
                bootidx_oob.append(bootidx_oob_i)

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1

            if y.ndim > 1:
                y_nvars = y.shape[1]
            else:
                y_nvars = 1
                y = y.reshape(-1, 1)

            # Initialize list structures to contain the fit
            boot_loadings_p = numpy.zeros((nboots, x_nvars, self.ncomps))
            boot_loadings_q = numpy.zeros((nboots, y_nvars, self.ncomps))
            boot_weights_w = numpy.zeros((nboots, x_nvars, self.ncomps))
            boot_weights_c = numpy.zeros((nboots, y_nvars, self.ncomps))
            boot_train_scores_t = list()
            boot_train_scores_u = list()

            # CV test scores more informative for ShuffleSplit than KFold but kept here anyway
            boot_test_scores_t = list()
            boot_test_scores_u = list()

            boot_rotations_ws = numpy.zeros((nboots, x_nvars, self.ncomps))
            boot_rotations_cs = numpy.zeros((nboots, y_nvars, self.ncomps))
            boot_betacoefs = numpy.zeros((nboots, y_nvars, x_nvars))
            boot_vipsw = numpy.zeros((nboots, y_nvars, x_nvars))

            # Initialise predictive residual sum of squares variable (for whole CV routine)
            pressy = 0
            pressx = 0

            # Calculate Sum of Squares SS in whole dataset for future calculations
            ssx = numpy.sum(numpy.square(bootstrap_pipeline.x_scaler.fit_transform(x)))
            ssy = numpy.sum(numpy.square(bootstrap_pipeline.y_scaler.fit_transform(y)))

            # As assessed in the test set..., opposed to PRESS
            R2X_training = numpy.zeros(nboots)
            R2Y_training = numpy.zeros(nboots)
            # R2X and R2Y assessed in the test set
            R2X_test = numpy.zeros(nboots)
            R2Y_test = numpy.zeros(nboots)

            # Cross validation
            for boot_i, idx_i, idx_oob_i in zip(range(nboots), bootidx, bootidx_oob):
            
                # split the data explicitly
                train = idx_i
                test = idx_oob_i

                # Check dimensions for the indexing
                if y_nvars == 1:
                    ytrain = y[train]
                    ytest = y[test]
                else:
                    ytrain = y[train, :]
                    ytest = y[test, :]
                if x_nvars == 1:
                    xtrain = x[train]
                    xtest = x[test]
                else:
                    xtrain = x[train, :]
                    xtest = x[test, :]

                # Prepare the scaled X and Y test data
                # If testset_scale is True, these are scaled individually...

                # Comply with the sklearn scaler behaviour
                if ytest.ndim == 1:
                    ytest = ytest.reshape(-1, 1)
                    ytrain = ytrain.reshape(-1, 1)
                if xtest.ndim == 1:
                    xtest = xtest.reshape(-1, 1)
                    xtrain = xtrain.reshape(-1, 1)

                bootstrap_pipeline.fit(xtrain, ytrain, **bootstrap_kwargs)

                # Prepare the scaled X and Y test data

                # Comply with the sklearn scaler behaviour
                if xtest.ndim == 1:
                    xtest = xtest.reshape(-1, 1)
                    xtrain = xtrain.reshape(-1, 1)
                # Fit the training data
                xtest_scaled = bootstrap_pipeline.x_scaler.transform(xtest)
                ytest_scaled = bootstrap_pipeline.y_scaler.transform(ytest)

                R2X_training[boot_i] = bootstrap_pipeline.score(xtrain, ytrain, 'x')
                R2Y_training[boot_i] = bootstrap_pipeline.score(xtrain, ytrain, 'y')
                
                # Use super here  for Q2
                ypred = bootstrap_pipeline.predict(x=xtest, y=None)
                xpred = bootstrap_pipeline.predict(x=None, y=ytest)

                xpred = bootstrap_pipeline.x_scaler.transform(xpred).squeeze()
                ypred = bootstrap_pipeline.y_scaler.transform(ypred).squeeze()
                ytest_scaled = ytest_scaled.squeeze()

                curr_pressx = numpy.sum(numpy.square(xtest_scaled - xpred))
                curr_pressy = numpy.sum(numpy.square(ytest_scaled - ypred))

                R2X_test[boot_i] = cv_pipeline.score(xtest, ytest, 'x')
                R2Y_test[boot_i] = cv_pipeline.score(xtest, ytest, 'y')

                pressx += curr_pressx
                pressy += curr_pressy

                boot_loadings_p[boot_i, :, :] = bootstrap_pipeline.loadings_p
                boot_loadings_q[boot_i, :, :] = bootstrap_pipeline.loadings_q
                boot_weights_w[boot_i, :, :] = bootstrap_pipeline.weights_w
                boot_weights_c[boot_i, :, :] = bootstrap_pipeline.weights_c
                boot_rotations_ws[boot_i, :, :] = bootstrap_pipeline.rotations_ws
                boot_rotations_cs[boot_i, :, :] = bootstrap_pipeline.rotations_cs
                boot_betacoefs[boot_i, :, :] = bootstrap_pipeline.beta_coeffs.T
                boot_vipsw[boot_i, :, :] = bootstrap_pipeline.VIP().T


            # Do a proper investigation on how to get CV scores decently
            # Align model parameters to account for sign indeterminacy.
            # The criteria here used is to select the sign that gives a more similar profile (by L1 distance) to the loadings from
            # on the model fitted with the whole data. Any other parameter can be used, but since the loadings in X capture
            # the covariance structure in the X data block, in theory they should have more pronounced features even in cases of
            # null X-Y association, making the sign flip more resilient.
            for boot_i in range(0, nboots):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = numpy.argmin(
                        numpy.array([numpy.sum(numpy.abs(self.loadings_p[:, currload] - boot_loadings_p[boot_i, :, currload])),
                                  numpy.sum(numpy.abs(
                                      self.loadings_p[:, currload] - boot_loadings_p[boot_i, :, currload] * -1))]))
                    if choice == 1:
                        boot_loadings_p[boot_i, :, currload] = -1 * boot_loadings_p[boot_i, :, currload]
                        boot_loadings_q[boot_i, :, currload] = -1 * boot_loadings_q[boot_i, :, currload]
                        boot_weights_w[boot_i, :, currload] = -1 * boot_weights_w[boot_i, :, currload]
                        boot_weights_c[boot_i, :, currload] = -1 * boot_weights_c[boot_i, :, currload]
                        boot_rotations_ws[boot_i, :, currload] = -1 * boot_rotations_ws[boot_i, :, currload]
                        boot_rotations_cs[boot_i, :, currload] = -1 * boot_rotations_cs[boot_i, :, currload]
                        boot_train_scores_t.append([*zip(train, -1 * bootstrap_pipeline.scores_t)])
                        boot_train_scores_u.append([*zip(train, -1 * bootstrap_pipeline.scores_u)])
                        boot_test_scores_t.append([*zip(test, -1 * bootstrap_pipeline.scores_t)])
                        boot_test_scores_u.append([*zip(test, -1 * bootstrap_pipeline.scores_u)])
                    else:
                        boot_train_scores_t.append([*zip(train, bootstrap_pipeline.scores_t)])
                        boot_train_scores_u.append([*zip(train, bootstrap_pipeline.scores_u)])
                        boot_test_scores_t.append([*zip(test, bootstrap_pipeline.scores_t)])
                        boot_test_scores_u.append([*zip(test, bootstrap_pipeline.scores_u)])

            # Calculate Q-squareds
            q_squaredy = 1 - (pressy / ssy)
            q_squaredx = 1 - (pressx / ssx)

            # Store everything...
            self.bootstrapParameters = {
                'Q2X': q_squaredx, 'Q2Y': q_squaredy,
                'MeanR2X_Training': numpy.mean(R2X_training), 'StdevR2X_Training': numpy.std(R2X_training),
                'MeanR2Y_Training': numpy.mean(R2Y_training), 'StdevR2Y_Training': numpy.std(R2X_training),
                'MeanR2X_Test': numpy.mean(R2X_test), 'StdevR2X_Test': numpy.std(R2X_test),
                'MeanR2Y_Test': numpy.mean(R2Y_test), 'StdevR2Y_Test': numpy.std(R2Y_test),
                    }

            # Save everything found during CV
            if outputdist is True:
                self.bootstrapParameters['bootR2X_Training'] = R2X_training
                self.bootstrapParameters['bootR2Y_Training'] = R2Y_training
                self.bootstrapParameters['bootR2X_Test'] = R2X_test
                self.bootstrapParameters['bootR2Y_Test'] = R2Y_test
                self.bootstrapParameters['boot_Loadings_q'] = boot_loadings_q
                self.bootstrapParameters['boot_Loadings_p'] = boot_loadings_p
                self.bootstrapParameters['boot_Weights_c'] = boot_weights_c
                self.bootstrapParameters['boot_Weights_w'] = boot_weights_w
                self.bootstrapParameters['boot_Rotations_ws'] = boot_rotations_ws
                self.bootstrapParameters['boot_Rotations_cs'] = boot_rotations_cs
                self.bootstrapParameters['boot_TestScores_t'] = boot_test_scores_t
                self.bootstrapParameters['boot_TestScores_u'] = boot_test_scores_u
                self.bootstrapParameters['boot_TrainScores_t'] = boot_train_scores_t
                self.bootstrapParameters['boot_TrainScores_u'] = boot_train_scores_u
                self.bootstrapParameters['boot_Beta'] = boot_betacoefs
                self.bootstrapParameters['boot_VIPw'] = boot_vipsw

            return None

        except ValueError as exp:
            raise exp

    def _cummulativefit(self, x, y):
        """
        Cumulative Regression Sum of Squares.

        Measure the cumulative Regression Sum of Squares for each individual component.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.

        Returns
        -------
        cumulative_fit : dict
            Dictionary containing the total Regression Sum of Squares and the Sum of Squares per components, for both the X and Y data blocks.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self._isfitted is False:
            raise AttributeError('fit model first')

        xscaled = self.x_scaler.transform(x)
        yscaled = self.y_scaler.transform(y)

        # Obtain residual sum of squares for whole data set and per component
        SSX = numpy.sum(numpy.square(xscaled))
        SSY = numpy.sum(numpy.square(yscaled))
        ssx_comp = list()
        ssy_comp = list()

        for curr_comp in range(1, self.ncomps + 1):
            model = self._reduce_ncomps(curr_comp)

            ypred = self.y_scaler.transform(ChemometricsPLS.predict(model, x, y=None))
            xpred = self.x_scaler.transform(ChemometricsPLS.predict(model, x=None, y=y))
            rssy = numpy.sum(numpy.square(yscaled - ypred))
            rssx = numpy.sum(numpy.square(xscaled - xpred))
            ssx_comp.append(rssx)
            ssy_comp.append(rssy)

        cumulative_fit = {'SSX': SSX, 'SSY': SSY, 'SSXcomp': numpy.array(ssx_comp), 'SSYcomp': numpy.array(ssy_comp)}

        return cumulative_fit

    def _reduce_ncomps(self, ncomps):
        """
        Reduce components.

        Generate a new model with a smaller set of components.

        Parameters
        ----------
        ncomps : int
            Number of ordered first N components from the original model to be kept.
            Must be smaller than the ncomps value of the original model.
    
        Returns
        -------
        newmodel : ChemometricsPLS
            ChemometricsPLS object with reduced number of components.

        Raises
        ------
        AttributeError
            If model is not fitted.
        ValueError
            If number of components desired is larger than original number of components
        """
        try:
            if ncomps > self.ncomps:
                raise ValueError('Fit a new model with more components instead')
            if self._isfitted is False:
                raise AttributeError('Model not Fitted')

            newmodel = deepcopy(self)
            newmodel._ncomps = ncomps

            newmodel.modelParameters = None
            newmodel.cvParameters = None
            newmodel.loadings_p = self.loadings_p[:, 0:ncomps]
            newmodel.weights_w = self.weights_w[:, 0:ncomps]
            newmodel.weights_c = self.weights_c[:, 0:ncomps]
            newmodel.loadings_q = self.loadings_q[:, 0:ncomps]
            newmodel.rotations_ws = self.rotations_ws[:, 0:ncomps]
            newmodel.rotations_cs = self.rotations_cs[:, 0:ncomps]
            newmodel.scores_t = None
            newmodel.scores_u = None
            newmodel.b_t = self.b_t[0:ncomps, 0:ncomps]
            newmodel.b_u = self.b_u[0:ncomps, 0:ncomps]

            # These have to be recalculated from the rotations
            newmodel.beta_coeffs = numpy.dot(newmodel.rotations_ws, newmodel.loadings_q.T)

            return newmodel
        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class ChemometricsPLSDA(ChemometricsPLS, ClassifierMixin):
    """
    PLS-DA object.
    
    Chemometrics PLS-DA object - Similar to ChemometricsPLS, but with extra functions to handle
    Y vectors encoding class membership and classification assessment metrics.
    PLS-DA (y with dummy matrix), where the predicted y from PLS response is converted into a class membership 
    prediction using the scores and a simple rule (proximity to class centroid in score space, calculated in training).
    
    The underlying PLS model is exactly the same as standard PLS, and this objects inherits from ChemometricsPLS, the main 
    difference in the PLS algorithm is the automated generation of dummy Y matrices. 
    
    Model performance metrics employed are the Q2Y, Area under the curve and ROC curves, f1 measure, balanced accuracy,
    precision, recall, confusion matrices and 0-1 loss. Although the Q2Y is seen, specific metrics for
    classification problems are recommended.    

    Parameters
    ----------
    ncomps : int
        Number of PLS components.
    pls_algorithm : class
        scikit-learn PLS algorithm to use - PLSRegression or PLSCanonical are supported.
    xscaler : CustomScaler object
        Scaler object for X data matrix.
    yscaler : CustomScaler object
        Scaler object for y data matrix.
    kwargs : pls_type_kwargs
        Keyword arguments to be passed during initialization of pls_algorithm.
    
    Raises
    ------
    TypeError
        If the pls_algorithm or scaler objects are not of the right class.
    
    Notes
    -----
    [1] Indhal et. al., From dummy regression to prior probabilities in PLS-DA, Journal of Chemometrics, 2007
    [2] Barker, Matthew, Rayens, William, Partial least squares for discrimination, Journal of Chemometrics, 2003
    [3] Brereton, Richard G. Lloyd, Gavin R., Partial least squares discriminant analysis: Taking the magic away, Journal of Chemometrics, 2014
    """

    def __init__(self, ncomps=2, pls_algorithm=PLSRegression,
                 xscaler=CustomScalerAuto(scale_power=1), **pls_type_kwargs):
        """
        :param ncomps:
        :param pls_algorithm:
        :param logreg_algorithm:
        :param xscaler:
        :param pls_type_kwargs:
        """
        try:
            # Perform the check with is instance but avoid abstract base class runs.
            pls_algorithm = pls_algorithm(ncomps, scale=False, **pls_type_kwargs)
            if not isinstance(pls_algorithm, (BaseEstimator)):
                raise TypeError("Scikit-learn model please")
            if not (isinstance(xscaler, TransformerMixin) or xscaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")

            # 2 blocks of data = two scaling options in PLS but here...
            if xscaler is None:
                xscaler = CustomScalerAuto(scale_power = 0, with_std=False)

            # Secretly declared here so calling methods from parent ChemometricsPLS class is possible
            self._y_scaler = CustomScalerResponse(scale_power = 0, with_std=False, with_mean=True)
            # Force y_scaling scaling to false, as this will be handled by the provided scaler or not
            # in PLS_DA/Logistic/LDA the y scaling is not used anyway,
            # but the interface is respected nevertheless

            self.pls_algorithm = pls_algorithm
            self.x_raw = None
            self.y_raw = None
            self.y_raw_unique = None
            # Most initialized as None, before object is fitted...
            self.scores_t = None
            self.scores_u = None
            self.weights_w = None
            self.weights_c = None
            self.loadings_p = None
            self.loadings_q = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.b_u = None
            self.b_t = None
            self.beta_coeffs = None
            self.n_classes = None
            self.class_means = None
            self._ncomps = ncomps
            self._x_scaler = xscaler
            self.cvParameters = None
            self.modelParameters = None
            self.permutationParameters = None
            self.bootstrapParameters = None
            self._isfitted = False

        except TypeError as terp:
            print(terp.args[0])

    def fit(self, x, y, **fit_params):
        """
        Model fit.

        Perform model fitting on the provided X data matrix and calculate basic goodness-of-fit metrics.
        Similar to scikit-learn's default BaseEstimator method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to fit the PLS model.
        y : numpy.ndarray, shape [n_samples, n_responses]
            Response matrix to fit the PLS model.
        kwargs : fit_params
            Keyword arguments to be passed to the .fit() method of the core sklearn model.
        
        Raises
        ------
        ValueError
            If any problem occurs during fitting.
        """
        try:
            # Save data
            self.x_raw = x
            self.y_raw = y
            self.y_raw_unique = [item for item in pandas.Series(y).unique()]
            # Not so important as don't expect a user applying a single x variable to a multivariate regression
            # method, but for consistency/testing purposes
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            # Scaling for the classifier setting proceeds as usual for the X block
            xscaled = self.x_scaler.fit_transform(x)

            # For this "classifier" PLS objects, the yscaler is not used, as we are not interesting in decentering and
            # scaling class labels and dummy matrices.

            # Instead, we just do some on the fly detection of binary vs multiclass classification
            # Verify number of classes in the provided class label y vector so the algorithm can adjust accordingly
            n_classes = numpy.unique(y).size
            self.n_classes = n_classes

            # If there are more than 2 classes, a Dummy 0-1 matrix is generated so PLS can do its job in
            # multi-class setting
            # Only for PLS: the sklearn LogisticRegression still requires a single vector!
            if self.n_classes > 2:
                dummy_mat = pandas.get_dummies(y).values
                # If the user wants OneVsRest, etc, provide a different binary labelled y vector to use it instead.
                y_scaled = self.y_scaler.fit_transform(dummy_mat)
            else:
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                y_scaled = self.y_scaler.fit_transform(y)

            # The PLS algorithm either gets a single vector in binary classification or a
            # Dummy matrix for the multiple classification case

            self.pls_algorithm.fit(xscaled, y_scaled, **fit_params)

            # Expose the model parameters - Same as in ChemometricsPLS
            self.loadings_p = self.pls_algorithm.x_loadings_
            self.loadings_q = self.pls_algorithm.y_loadings_
            self.weights_w = self.pls_algorithm.x_weights_
            self.weights_c = self.pls_algorithm.y_weights_
            self.rotations_ws = self.pls_algorithm.x_rotations_
            # scikit learn sets the rotation, causing a discrepancy between the scores calculated during fitting and the transform method
            # for now, we calculate the rotation and override it: C* = pinv(CQ')C
            self.rotations_cs = numpy.dot(numpy.linalg.pinv(numpy.dot(self.weights_c, self.loadings_q.T)), self.weights_c)
            self.scores_t = self.pls_algorithm.x_scores_
            self.scores_u = self.pls_algorithm.y_scores_
            self.b_u = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(self.scores_u.T, self.scores_u)), self.scores_u.T),
                              self.scores_t)
            self.b_t = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(self.scores_t.T, self.scores_t)), self.scores_t.T),
                              self.scores_u)
            self.beta_coeffs = self.pls_algorithm.coef_

            # Get the mean score per class to use in prediction
            # To use im a simple rule on how to turn PLS prediction into a classifier for multiclass PLS-DA
            self.class_means = numpy.zeros((n_classes, self.ncomps))
            for curr_class in self.y_raw_unique:
                curr_class_idx = numpy.where(y == curr_class)
                self.class_means[curr_class, :] = numpy.mean(self.scores_t[curr_class_idx])

            # Needs to come here for the method shortcuts down the line to work...
            self._isfitted = True

            # Calculate RSSy/RSSx, R2Y/R2X
            # Method inheritance from parent, as in this case we really want the "PLS" only metrics
            if self.n_classes > 2:
                R2Y = ChemometricsPLS.score(self, x=x, y=dummy_mat, block_to_score='y')
                R2X = ChemometricsPLS.score(self, x=x, y=dummy_mat, block_to_score='x')
            else:
                R2Y = ChemometricsPLS.score(self, x=x, y=y, block_to_score='y')
                R2X = ChemometricsPLS.score(self, x=x, y=y, block_to_score='x')
            

            # Obtain the class score
            class_score = ChemometricsPLS.predict(self, x=x)
            
            if self.n_classes == 2:
                roc_fpr = numpy.zeros((len(y),1))
                roc_fpr[:] = numpy.nan
                roc_tpr = numpy.zeros((len(y),1))
                roc_tpr[:] = numpy.nan
                auc = numpy.zeros((1,1))
            else:
                roc_fpr = numpy.zeros((len(y),self.n_classes))
                roc_fpr[:] = numpy.nan
                roc_tpr = numpy.zeros((len(y),self.n_classes))
                roc_tpr[:] = numpy.nan
                auc = numpy.zeros((self.n_classes,))

            if self.n_classes == 2:
                y_pred = self.predict(x)
                accuracy = metrics.accuracy_score(y, y_pred)
                precision = metrics.precision_score(y, y_pred)
                recall = metrics.recall_score(y, y_pred)
                misclassified_samples = numpy.where(y.ravel() != y_pred.ravel())[0]
                f1_score = metrics.f1_score(y, y_pred)
                conf_matrix = metrics.confusion_matrix(y, y_pred)
                zero_oneloss = metrics.zero_one_loss(y, y_pred)
                matthews_mcc = metrics.matthews_corrcoef(y, y_pred)

                # Interpolated ROC curve and AUC
                fpr, tpr, _ = metrics.roc_curve(y, class_score.ravel())
                roc_fpr[:len(fpr),0] = fpr
                roc_tpr[:len(tpr),0] = tpr
                auc[0] = metrics.auc(fpr, tpr)

            else:
                y_pred = self.predict(x)
                accuracy = metrics.accuracy_score(y, y_pred)
                precision = metrics.precision_score(y, y_pred, average='weighted')
                recall = metrics.recall_score(y, y_pred, average='weighted')
                misclassified_samples = numpy.where(y.ravel() != y_pred.ravel())[0]
                f1_score = metrics.f1_score(y, y_pred, average='weighted')
                conf_matrix = metrics.confusion_matrix(y, y_pred)
                zero_oneloss = metrics.zero_one_loss(y, y_pred)
                matthews_mcc = numpy.nan

                # Generate multiple ROC curves - one for each class the multiple class case
                for predclass in self.y_raw_unique:
                    fpr, tpr, _ = metrics.roc_curve(y, class_score[:, predclass], pos_label=predclass)
                    roc_fpr[:len(fpr),predclass] = fpr
                    roc_tpr[:len(tpr),predclass] = tpr
                    auc[predclass] = metrics.auc(fpr, tpr)

            # Obtain residual sum of squares for whole data set and per component
            # Same as Chemometrics PLS, this is so we can use VIP's and other metrics as usual
            if self.n_classes > 2:
                cm_fit = self._cummulativefit(x, dummy_mat)
            else:
                cm_fit = self._cummulativefit(x, y)

            # Assemble the dictionary for storing the model parameters
            self.modelParameters = {'PLS': {'R2Y': R2Y, 'R2X': R2X, 'SSX': cm_fit['SSX'], 'SSY': cm_fit['SSY'],
                                            'SSXcomp': cm_fit['SSXcomp'], 'SSYcomp': cm_fit['SSYcomp']},
                                    'DA': {'Accuracy': accuracy, 'AUC': auc,
                                                 'ConfusionMatrix': conf_matrix, 'ROC_fpr': roc_fpr, 'ROC_tpr': roc_tpr,
                                                 'MisclassifiedSamples': misclassified_samples,
                                                 'Precision': precision, 'Recall': recall,
                                                 'F1': f1_score, '0-1Loss': zero_oneloss, 'MatthewsMCC': matthews_mcc,
                                                 'ClassPredictions': y_pred}}

        except ValueError as verr:
            raise verr

    def fit_transform(self, x, y, **fit_params):
        """
        Model fit and transform data.

        Fit a model and return the scores, as per the scikit-learn's TransformerMixin method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to fit the PLS model.
        y : numpy.ndarray, shape [n_samples, n_responses]
            Response matrix to fit the PLS model.
        kwargs : fit_params
            Keyword arguments to be passed to the .fit() method of the core sklearn model.
        
        Returns
        -------
        (T,U) : tuple of numpy.ndarray, shape [[n_tscores], [n_uscores]]
            Latent Variable scores (T) for the X matrix and for the Y vector/matrix (U).

        Raises
        ------
        ValueError
            If there are problems with the input or during model fitting.
        """

        try:
            # Self explanatory - the scaling and sorting out of the Y vector will be handled inside
            self.fit(x, y, **fit_params)
            return self.transform(x, y=None), self.transform(x=None, y=y)

        except ValueError as verr:
            raise verr

    def transform(self, x=None, y=None):
        """
        Transform data.

        Calculate the scores for a data block from the original data. Equivalent to sklearn's TransformerMixin method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix to fit the PLS model.
        y : numpy.ndarray, shape [n_samples, n_responses]
            Response matrix to fit the PLS model.
        
        Returns
        -------
        (T,U) : tuple of numpy.ndarray, shape [[n_tscores], [n_uscores]]
            Latent Variable scores (T) for the X matrix and for the Y vector/matrix (U).

        Raises
        ------
        ValueError
            If dimensions of input data are mismatched.
        AttributeError
            When calling the method before the model is fitted.
        """
        try:
            # Check if model is fitted
            if self._isfitted is True:
                # If X and Y are passed, complain and do nothing
                if (x is not None) and (y is not None):
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif (x is None) and (y is None):
                    raise ValueError('yy')
                # If Y is given, return U
                elif x is None:
                    # The y variable expected is a single vector with ints as class label - binary
                    # and multiclass classification are allowed but not multilabel so this will work.
                    if y.ndim != 1:
                        raise TypeError('Please supply a dummy vector with integer as class membership')

                    # Previously fitted model will already have the number of classes
                    # The dummy matrix is created here manually because its possible for the model to be fitted to
                    # a larger number of classes than what is being passed in transform
                    # and other methods post-fitting
                    # If matrix is not dummy, generate the dummy accordingly
                    if self.n_classes > 2:
                        y = self.y_scaler.transform(pandas.get_dummies(y).values)
                    else:
                        if y.ndim == 1:
                            y = y.reshape(-1, 1)
                        y = self.y_scaler.transform(y)
                    # Taking advantage of rotations_y
                    # Otherwise this would be the full calculation U = Y*pinv(CQ')*C
                    U = numpy.dot(y, self.rotations_cs)
                    return U

                # If X is given, return T
                elif y is None:
                    # Comply with the sklearn scaler behaviour and X scaling - business as usual
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    xscaled = self.x_scaler.transform(x)
                    # Taking advantage of the rotation_x
                    # Otherwise this would be would the full calculation T = X*pinv(WP')*W
                    T = numpy.dot(xscaled, self.rotations_ws)
                    return T
            else:
                raise AttributeError('Model not fitted')
        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    def inverse_transform(self, t=None, u=None):
        """
        Inverse transform data.

        Transform scores to the original data space using their corresponding loadings.
        Similar to scikit-learn's default TransformerMixin method.

        Parameters
        ----------
        t : numpy.ndarray, shape [n_samples, n_comps] or None
            T scores corresponding to the X data matrix.
        u : numpy.ndarray, shape [n_samples, n_responses] or None
            U scores corresponding to the y data matrix.
        
        Returns
        -------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.

        Raises
        ------
        ValueError
            If dimensions of input data are mismatched.
        """
        try:
            if self._isfitted is True:
                if t is not None and u is not None:
                    raise ValueError('xx')
                # If nothing is passed at all, complain and do nothing
                elif t is None and u is None:
                    raise ValueError('yy')
                # If T is given, return U
                elif t is not None:
                    # Calculate X from T using X = TP'
                    xpred = numpy.dot(t, self.loadings_p.T)
                    if self.x_scaler is not None:
                        xscaled = self.x_scaler.inverse_transform(xpred)
                    else:
                        xscaled = xpred

                    return xscaled

                # If U is given, return T
                # This might be a bit weird in dummy matrices/etc, but kept here for "symmetry" with
                # parent ChemometricsPLS implementation
                elif u is not None:
                    # Calculate Y from U - using Y = UQ'
                    ypred = numpy.dot(u, self.loadings_q.T)
                    return ypred

        except ValueError as verr:
            raise verr

    def score(self, x, y, sample_weight=None):
        """
        Score model.

        Predict and calculate the R2 for the model using one of the data blocks (X or Y) provided.
        Equivalent to the scikit-learn RegressorMixin score method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.
        block_to_score : str
            Which of the data blocks (X or Y) to calculate the R2 goodness of fit.
        sample_weight : numpy.ndarray, shape [n_samples] or None
            Optional sample weights to use in scoring.
        
        Returns
        -------
        R2Y : float
            The model's R2Y, calculated by predicting Y from X and scoring.
        R2X : float
            The model's R2X, calculated by predicting X from Y and scoring.

        Raises
        ------
        ValueError
            If block to score argument is not acceptable or date mismatch issues with the provided data.
        """
        try:
            return metrics.accuracy_score(y, self.predict(x), sample_weight=sample_weight)
        except ValueError as verr:
            raise verr

    def predict(self, x):
        """
        Predict scores.

        Predict the values in one data block using the other. Same as its scikit-learn's RegressorMixin namesake method.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.
       
        Returns
        -------
        predicted : numpy.ndarray, shape [n_samples, n_features] or shape [n_samples, n_responses]
            Predicted data block (X or Y) obtained from the other data block.

        Raises
        ------
        ValueError
            If no data matrix is passed, or dimensions mismatch issues with the provided data.
        AttributeError
            Calling the method without fitting the model before.
        """
        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")

            # based on original encoding as 0, 1, etc
            if self.n_classes == 2:
                y_pred = ChemometricsPLS.predict(self, x)
                class_pred = numpy.argmin(numpy.abs(y_pred - numpy.array([0, 1])), axis=1)

            else:
                # euclidean distance to mean of class for multiclass PLS-DA
                # probably better to use a Logistic/Multinomial or PLS-LDA anyway...
                # project X onto T - so then we can get
                pred_scores = self.transform(x=x)
                # prediction rule - find the closest class mean (centroid) for each sample in the score space
                closest_class_mean = lambda x: numpy.argmin(numpy.linalg.norm((x - self.class_means), axis=1))
                class_pred = numpy.apply_along_axis(closest_class_mean, axis=1, arr=pred_scores)
            return class_pred

        except ValueError as verr:
            raise verr
        except AttributeError as atter:
            raise atter

    @property
    def ncomps(self):
        try:
            return self._ncomps
        except AttributeError as atre:
            raise atre

    @ncomps.setter
    def ncomps(self, ncomps=1):
        """
        Setter for number of components.

        Parameters
        ----------
        ncomps : int
            Number of PLS components to use in the model.

        Raises
        ------
        AttributeError
            If there is a problem changing the number of components and resetting the model.
        """
        # To ensure changing number of components effectively resets the model
        try:
            self._ncomps = ncomps
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.pls_algorithm.n_components = ncomps
            self.x_raw = None
            self.y_raw = None
            self.y_raw_unique = None
            self.loadings_p = None
            self.scores_t = None
            self.scores_u = None
            self.loadings_q = None
            self.weights_c = None
            self.weights_w = None
            self.rotations_cs = None
            self.rotations_ws = None
            self.cvParameters = None
            self.modelParameters = None
            self.permutationParameters = None
            self.bootstrapParameters = None
            self.b_t = None
            self.b_u = None
            self.beta_coeffs = None
            self.n_classes = None

            return None

        except AttributeError as atre:
            raise atre

    @property
    def x_scaler(self):
        try:
            return self._x_scaler
        except AttributeError as atre:
            raise atre

    @x_scaler.setter
    def x_scaler(self, scaler):
        """
        Setter for the X data block scaler.

        Parameters
        ----------
        scaler : CustomScaler object
            Scaling/preprocessing object or None.

        Raises
        ------
        AttributeError
            If there is a problem changing the scaler and resetting the model.
        TypeError
            If the new scaler provided is not a valid object.
        """
        try:

            if not (isinstance(scaler, TransformerMixin) or scaler is None):
                raise TypeError("Scikit-learn Transformer-like object or None")
            if scaler is None:
                scaler = CustomScalerAuto(scale_power = 0, with_std=False)

            self._x_scaler = scaler
            self.pls_algorithm = clone(self.pls_algorithm, safe=True)
            self.x_raw = None
            self.y_raw = None
            self.y_raw_unique = None
            self.modelParameters = None
            self.cvParameters = None
            self.permutationParameters = None
            self.bootstrapParameters = None
            self.loadings_p = None
            self.weights_w = None
            self.weights_c = None
            self.loadings_q = None
            self.rotations_ws = None
            self.rotations_cs = None
            self.scores_t = None
            self.scores_u = None
            self.b_t = None
            self.b_u = None
            self.beta_coeffs = None
            self.n_classes = None

            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    @property
    def y_scaler(self):
        try:
            return self._y_scaler
        except AttributeError as atre:
            raise atre

    @y_scaler.setter
    def y_scaler(self, scaler):
        """
        Setter for the Y data block scaler.

        Parameters
        ----------
        scaler : CustomScaler object
            Scaling/preprocessing object or None.

        Raises
        ------
        AttributeError
            If there is a problem changing the scaler and resetting the model.
        TypeError
            If the new scaler provided is not a valid object.
        """
        try:
            # ignore the value -
            self._y_scaler = CustomScalerResponse(scale_power = 0, with_std=False, with_mean=True)
            return None

        except AttributeError as atre:
            raise atre
        except TypeError as typerr:
            raise typerr

    def VIP(self):
        """
        Variable Importance on Projection.

        Output the Variable importance for projection metric (VIP). 
        With the default values it is calculated using the x variable weights and the variance explained of y. 
        Default mode is recommended (mode = 'w' and direction = 'y').
        
        Parameters
        ----------
        mode : str
            The type of model parameter to use in calculating the VIP. 
            Default value is weights (w), and other acceptable arguments are p, ws, cs, c and q.
        direction : str
            The data block to be used to calculated the model fit and regression sum of squares.
        
        Returns
        -------
        VIP : numpy.ndarray, shape [n_features]
            The vector with the calculated VIP values.

        Raises
        ------
        ValueError
            If mode or direction is not a valid option.
        AttributeError
            Calling method without a fitted model.
        """
        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")

            y = self.y_raw
            y_unique = self.y_raw_unique
            # Get data
            t = self.scores_t
            w = self.weights_w
            q = self.loadings_q
            p, h = w.shape
            # Preallocate array
            vips = numpy.zeros((p,q.shape[0]))
            # Multi-class VIP
            if self.n_classes > 2:
                # Cycle classes
                for k in y_unique:
                    vips_k = numpy.zeros((p,))
                    # SSY and SSYcum is different
                    ind_k = numpy.where(y==k)[0]
                    t_k = t[ind_k,:]
                    q_k = q
                    # Calculate SSY and SSYcum
                    SSY = numpy.diag(t_k.T @ t_k @ q_k.T @ q_k).reshape(h, -1)
                    SSYcum = numpy.sum(SSY)
                    for i in range(p):
                        weight = numpy.array([(w[i,j] / numpy.linalg.norm(w[:,j]))**2 for j in range(h)])
                        vips_k[i] = numpy.sqrt(p*(SSY.T @ weight)/SSYcum)
                    vips[:,k] = vips_k
            else:
                # Calculate SSY and SSYcum
                SSY = numpy.diag(t.T @ t @ q.T @ q).reshape(h, -1)
                SSYcum = numpy.sum(SSY)

                # Full model
                vips = numpy.zeros((p,))
                for i in range(p):
                    weight = numpy.array([(w[i,j] / numpy.linalg.norm(w[:,j]))**2 for j in range(h)])
                    vips[i] = numpy.sqrt(p*(SSY.T @ weight)/SSYcum)
            return vips
        except AttributeError as atter:
            raise atter
        except ValueError as verr:
            raise verr

    def hotelling_T2(self, comps=None, alpha=0.05):
        """
        Hotelling T2 ellipse.

        Obtain the parameters for the Hotelling T2 ellipse at the desired significance level.

        Parameters
        ----------
        comps : list
            List of components in 2D.
        alpha : float
            Significance level.
        
        Returns
        -------
        radii : numpy.ndarray
            The Hotelling T2 ellipsoid radii at vertex.

        Raises
        ------
        AtributeError
            If the model is not fitted.
        ValueError
            If the components requested are higher than the number of components in the model.
        TypeError
            If comps is not None or list/numpy 1d array and alpha a float.
        """
        try:
            if self._isfitted is False:
                raise AttributeError("Model is not fitted")

            nsamples = self.scores_t.shape[0]

            if comps is None:
                ncomps = self.ncomps
                ellips = self.scores_t[:, range(self.ncomps)] ** 2
            else:
                ncomps = len(comps)
                ellips = self.scores_t[:, comps] ** 2

            ellips = 1 / nsamples * (ellips.sum(0))

            # F stat
            a = (nsamples - 1) / nsamples * ncomps * (nsamples ** 2 - 1) / (nsamples * (nsamples - ncomps))
            a = a * st.f.ppf(1-alpha, ncomps, nsamples - ncomps)

            hoteling_t2 = list()
            for comp in range(ncomps):
                hoteling_t2.append(numpy.sqrt((a * ellips[comp])))

            return numpy.array(hoteling_t2)

        except AttributeError as atre:
            raise atre
        except ValueError as valerr:
            raise valerr
        except TypeError as typerr:
            raise typerr

    def cross_validation(self, x, y, cv_method=model_selection.KFold(7, shuffle=True), outputdist=False,
                         **crossval_kwargs):
        """
        Cross validation.

        Cross-validation method for the model. Calculates Q2 and cross-validated estimates for all model parameters.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.
        cv_method : BaseCrossValidator or BaseShuffleSplit
            An instance of a scikit-learn CrossValidator object.
        outputdist : bool
            Output the whole distribution for the cross validated parameters.
        crossval_kwargs : kwargs
            Keyword arguments to be passed to the sklearn.Pipeline during cross-validation.

        Returns
        -------
        cv_params : dict
            Adds a dictionary cvParameters to the object, containing the cross validation results.

        Raises
        ------
        TypeError
            If the cv_method passed is not a scikit-learn CrossValidator object.
        ValueError
            If the x and y data matrix is invalid.
        """
        try:
            # Check if global model is fitted... and if not, fit it using all of X
            if self._isfitted is False:
                self.fit(x, y)

            # Make a copy of the object, to ensure the internal state of the object is not modified during
            # the cross_validation method call
            cv_pipeline = deepcopy(self)
            # Number of splits
            ncvrounds = cv_method.get_n_splits()

            # Number of classes to select tell binary from multi-class discrimination parameter calculation
            n_classes = numpy.unique(y).size

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1

            # The y variable expected is a single vector with ints as class label - binary
            # and multiclass classification are allowed but not multilabel so this will work.
            # but for the PLS part in case of more than 2 classes a dummy matrix is constructed and kept separately
            # throughout
            if y.ndim == 1:
                # y = y.reshape(-1, 1)
                if self.n_classes > 2:
                    y_pls = pandas.get_dummies(y).values
                    y_nvars = y_pls.shape[1]
                else:
                    y_pls = y
                    y_nvars = 1
            else:
                raise TypeError('Please supply a dummy vector with integer as class membership')

            # Initialize list structures to contain the fit
            cv_loadings_p = numpy.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_loadings_q = numpy.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_weights_w = numpy.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_weights_c = numpy.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_train_scores_t = []
            cv_train_scores_u = []
            cv_test_scores_t = []
            cv_test_scores_u = []

            cv_rotations_ws = numpy.zeros((ncvrounds, x_nvars, self.ncomps))
            cv_rotations_cs = numpy.zeros((ncvrounds, y_nvars, self.ncomps))
            cv_betacoefs = numpy.zeros((ncvrounds, y_nvars, x_nvars))
            cv_vipsw = numpy.zeros((ncvrounds, y_nvars, x_nvars))

            cv_trainprecision = numpy.zeros(ncvrounds)
            cv_trainrecall = numpy.zeros(ncvrounds)
            cv_trainaccuracy = numpy.zeros(ncvrounds)
            cv_trainmatthews_mcc = numpy.zeros(ncvrounds)
            cv_trainzerooneloss = numpy.zeros(ncvrounds)
            cv_trainf1 = numpy.zeros(ncvrounds)
            cv_trainclasspredictions = list()
            cv_trainconfusionmatrix = list()
            cv_trainmisclassifiedsamples = list()

            cv_testprecision = numpy.zeros(ncvrounds)
            cv_testrecall = numpy.zeros(ncvrounds)
            cv_testaccuracy = numpy.zeros(ncvrounds)
            cv_testmatthews_mcc = numpy.zeros(ncvrounds)
            cv_testzerooneloss = numpy.zeros(ncvrounds)
            cv_testf1 = numpy.zeros(ncvrounds)
            cv_testclasspredictions = list()
            cv_testconfusionmatrix = list()
            cv_testmisclassifiedsamples = list()

            if self.n_classes == 2:
                cv_trainroc_fpr = numpy.zeros((ncvrounds, len(y),))
                cv_trainroc_fpr[:] = numpy.nan
                cv_trainroc_tpr = numpy.zeros((ncvrounds, len(y),))
                cv_trainroc_tpr[:] = numpy.nan
                cv_trainauc = numpy.zeros((ncvrounds,))

                cv_testroc_fpr = numpy.zeros((ncvrounds, len(y),))
                cv_testroc_fpr[:] = numpy.nan
                cv_testroc_tpr = numpy.zeros((ncvrounds, len(y),))
                cv_testroc_tpr[:] = numpy.nan
                cv_testauc = numpy.zeros((ncvrounds,))
            else:
                cv_trainroc_fpr = numpy.zeros((ncvrounds, len(y), y_nvars))
                cv_trainroc_fpr[:] = numpy.nan
                cv_trainroc_tpr = numpy.zeros((ncvrounds, len(y), y_nvars))
                cv_trainroc_tpr[:] = numpy.nan
                cv_trainauc = numpy.zeros((ncvrounds,y_nvars))

                cv_testroc_fpr = numpy.zeros((ncvrounds, len(y), y_nvars))
                cv_testroc_fpr[:] = numpy.nan
                cv_testroc_tpr = numpy.zeros((ncvrounds, len(y), y_nvars))
                cv_testroc_tpr[:] = numpy.nan
                cv_testauc = numpy.zeros((ncvrounds,y_nvars))

            # Initialise predictive residual sum of squares variable (for whole CV routine)
            pressy = 0
            pressx = 0

            # Calculate Sum of Squares SS in whole dataset for future calculations
            ssx = numpy.sum(numpy.square(cv_pipeline.x_scaler.fit_transform(x)))
            ssy = numpy.sum(numpy.square(cv_pipeline._y_scaler.fit_transform(y_pls.reshape(-1, 1))))

            # As assessed in the test set..., opposed to PRESS
            R2X_training = numpy.zeros(ncvrounds)
            R2Y_training = numpy.zeros(ncvrounds)
            # R2X and R2Y assessed in the test set
            R2X_test = numpy.zeros(ncvrounds)
            R2Y_test = numpy.zeros(ncvrounds)

            # Cross validation
            for cvround, train_testidx in enumerate(cv_method.split(x, y)):
                # split the data explicitly
                train = train_testidx[0]
                test = train_testidx[1]

                # Check dimensions for the indexing
                ytrain = y[train]
                ytest = y[test]
                if x_nvars == 1:
                    xtrain = x[train]
                    xtest = x[test]
                else:
                    xtrain = x[train, :]
                    xtest = x[test, :]

                cv_pipeline.fit(xtrain, ytrain, **crossval_kwargs)

                # Prepare the scaled X and Y test data

                # Comply with the sklearn scaler behaviour
                if xtest.ndim == 1:
                    xtest = xtest.reshape(-1, 1)
                    xtrain = xtrain.reshape(-1, 1)
                # Fit the training data

                xtest_scaled = cv_pipeline.x_scaler.transform(xtest)

                R2X_training[cvround] = ChemometricsPLS.score(cv_pipeline, xtrain, ytrain, 'x')
                R2Y_training[cvround] = ChemometricsPLS.score(cv_pipeline, xtrain, ytrain, 'y')
                
                if y_pls.ndim > 1:
                    yplstest = y_pls[test, :]

                else:
                    yplstest = y_pls[test].reshape(-1, 1)

                # Use super here  for Q2
                ypred = ChemometricsPLS.predict(cv_pipeline, x=xtest, y=None)
                xpred = ChemometricsPLS.predict(cv_pipeline, x=None, y=ytest)

                xpred = cv_pipeline.x_scaler.transform(xpred).squeeze()
                ypred = cv_pipeline._y_scaler.transform(ypred).squeeze()

                curr_pressx = numpy.sum(numpy.square(xtest_scaled - xpred))
                curr_pressy = numpy.sum(numpy.square(cv_pipeline._y_scaler.transform(yplstest).squeeze() - ypred))

                R2X_test[cvround] = ChemometricsPLS.score(cv_pipeline, xtest, yplstest, 'x')
                R2Y_test[cvround] = ChemometricsPLS.score(cv_pipeline, xtest, yplstest, 'y')

                pressx += curr_pressx
                pressy += curr_pressy

                ###
                cv_train_scores_t.append([(item[0], item[1]) for item in zip(train, cv_pipeline.scores_t)])
                cv_train_scores_u.append([(item[0], item[1]) for item in zip(train, cv_pipeline.scores_u)])

                pred_scores_t = cv_pipeline.transform(x = xtest, y = None)
                pred_scores_u = cv_pipeline.transform(x = None, y = ytest)

                cv_test_scores_t.append([(item[0], item[1]) for item in zip(test, pred_scores_t)])
                cv_test_scores_u.append([(item[0], item[1]) for item in zip(test, pred_scores_u)])
                ###

                cv_loadings_p[cvround, :, :] = cv_pipeline.loadings_p
                cv_loadings_q[cvround, :, :] = cv_pipeline.loadings_q
                cv_weights_w[cvround, :, :] = cv_pipeline.weights_w
                cv_weights_c[cvround, :, :] = cv_pipeline.weights_c
                cv_rotations_ws[cvround, :, :] = cv_pipeline.rotations_ws
                cv_rotations_cs[cvround, :, :] = cv_pipeline.rotations_cs
                cv_betacoefs[cvround, :, :] = cv_pipeline.beta_coeffs.T
                cv_vipsw[cvround, :, :] = cv_pipeline.VIP().T
                # Training metrics
                cv_trainaccuracy[cvround] = cv_pipeline.modelParameters['DA']['Accuracy']
                cv_trainprecision[cvround] = cv_pipeline.modelParameters['DA']['Precision']
                cv_trainrecall[cvround] = cv_pipeline.modelParameters['DA']['Recall']
                cv_trainf1[cvround] = cv_pipeline.modelParameters['DA']['F1']
                cv_trainmatthews_mcc[cvround] = cv_pipeline.modelParameters['DA']['MatthewsMCC']
                cv_trainzerooneloss[cvround] = cv_pipeline.modelParameters['DA']['0-1Loss']

                # Check this indexes, same as CV scores
                cv_trainmisclassifiedsamples.append(
                    train[cv_pipeline.modelParameters['DA']['MisclassifiedSamples']])
                cv_trainclasspredictions.append(
                    [*zip(train, cv_pipeline.modelParameters['DA']['ClassPredictions'])])

                # Predict y
                y_pred = cv_pipeline.predict(xtest)
                # Obtain the class score
                class_score = ChemometricsPLS.predict(cv_pipeline, xtest)

                # DA parameter
                fpr_grid = numpy.linspace(0, 1, num=20)
                if n_classes == 2:
                    cv_testaccuracy[cvround] = metrics.accuracy_score(ytest, y_pred)
                    cv_testprecision[cvround] = metrics.precision_score(ytest, y_pred)
                    cv_testrecall[cvround] = metrics.recall_score(ytest, y_pred)
                    cv_testf1[cvround]  = metrics.f1_score(ytest, y_pred)
                    cv_testzerooneloss[cvround] = metrics.zero_one_loss(ytest, y_pred)
                    cv_testmatthews_mcc[cvround] = metrics.matthews_corrcoef(ytest, y_pred)

                    # ROC
                    cv_trainroc_fpr[cvround, :len(cv_pipeline.modelParameters['DA']['ROC_fpr'].T[0])] = cv_pipeline.modelParameters['DA']['ROC_fpr'].T[0]
                    cv_trainroc_tpr[cvround, :len(cv_pipeline.modelParameters['DA']['ROC_fpr'].T[0])] = cv_pipeline.modelParameters['DA']['ROC_tpr'].T[0]
                    cv_trainauc[cvround] = cv_pipeline.modelParameters['DA']['AUC']

                    fpr, tpr, _ = metrics.roc_curve(ytest, class_score.ravel(), drop_intermediate = False)
                    cv_testroc_fpr[cvround, :len(fpr)] = fpr
                    cv_testroc_tpr[cvround, :len(tpr)] = tpr
                    cv_testauc[cvround] = metrics.auc(fpr, tpr)

                else:
                    cv_testaccuracy[cvround] = metrics.accuracy_score(ytest, y_pred)
                    cv_testprecision[cvround] = metrics.precision_score(ytest, y_pred, average='weighted')
                    cv_testrecall[cvround] = metrics.recall_score(ytest, y_pred, average='weighted')
                    cv_testf1[cvround] = metrics.f1_score(ytest, y_pred, average='weighted')
                    cv_testzerooneloss[cvround] = metrics.zero_one_loss(ytest, y_pred)
                    cv_testmatthews_mcc[cvround] = numpy.nan
                    
                    # Generate multiple ROC curves - one for each class the multiple class case
                    for predclass in range(cv_pipeline.n_classes):
                        # ROC
                        cv_trainroc_fpr[cvround, :len(cv_pipeline.modelParameters['DA']['ROC_fpr'][:, predclass]), predclass] = cv_pipeline.modelParameters['DA']['ROC_fpr'][:, predclass]
                        cv_trainroc_tpr[cvround, :len(cv_pipeline.modelParameters['DA']['ROC_tpr'][:, predclass]), predclass] = cv_pipeline.modelParameters['DA']['ROC_tpr'][:, predclass]
                        cv_trainauc[cvround, predclass] = cv_pipeline.modelParameters['DA']['AUC'][predclass]

                        fpr, tpr, _ = metrics.roc_curve(ytest, class_score[:, predclass], pos_label=predclass, drop_intermediate = False)
                        cv_testroc_fpr[cvround, :len(fpr), predclass] = fpr
                        cv_testroc_tpr[cvround, :len(tpr), predclass] = tpr
                        cv_testauc[cvround, predclass] = metrics.auc(fpr, tpr)

                # TODO check the roc curve in train and test set
                # Check the actual indexes in the original samples
                test_misclassified_samples = test[numpy.where(ytest.ravel() != y_pred.ravel())[0]]
                test_classpredictions = [*zip(test, y_pred)]
                test_conf_matrix = metrics.confusion_matrix(ytest, y_pred)

                # Check this indexes, same as CV scores
                cv_testmisclassifiedsamples.append(test_misclassified_samples)
                cv_testconfusionmatrix.append(test_conf_matrix)
                cv_testclasspredictions.append(test_classpredictions)

            # Do a proper investigation on how to get CV scores decently
            # Align model parameters to account for sign indeterminacy.
            # The criteria here used is to select the sign that gives a more similar profile (by L1 distance) to the loadings from
            # on the model fitted with the whole data. Any other parameter can be used, but since the loadings in X capture
            # the covariance structure in the X data block, in theory they should have more pronounced features even in cases of
            # null X-Y association, making the sign flip more resilient.
            for cvround in range(0, ncvrounds):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = numpy.argmin(
                        numpy.array([numpy.sum(numpy.abs(self.loadings_p[:, currload] - cv_loadings_p[cvround, :, currload])),
                                  numpy.sum(numpy.abs(
                                      self.loadings_p[:, currload] - cv_loadings_p[cvround, :, currload] * -1))]))

                    if choice == 1:
                        cv_loadings_p[cvround, :, currload] = -1 * cv_loadings_p[cvround, :, currload]
                        cv_loadings_q[cvround, :, currload] = -1 * cv_loadings_q[cvround, :, currload]
                        cv_weights_w[cvround, :, currload] = -1 * cv_weights_w[cvround, :, currload]
                        cv_weights_c[cvround, :, currload] = -1 * cv_weights_c[cvround, :, currload]
                        cv_rotations_ws[cvround, :, currload] = -1 * cv_rotations_ws[cvround, :, currload]
                        cv_rotations_cs[cvround, :, currload] = -1 * cv_rotations_cs[cvround, :, currload]

                        list_train_score_t = []
                        for idx, array in cv_train_scores_t[cvround]:
                            array[currload] = array[currload]*-1
                            list_train_score_t.append((idx,array))

                        list_train_score_u = []
                        for idx, array in cv_train_scores_u[cvround]:
                            array[currload] = array[currload]*-1
                            list_train_score_u.append((idx,array))

                        list_test_score_t = []
                        for idx, array in cv_test_scores_t[cvround]:
                            array[currload] = array[currload]*-1
                            list_test_score_t.append((idx,array))

                        list_test_score_u = []
                        for idx, array in cv_test_scores_u[cvround]:
                            array[currload] = array[currload]*-1
                            list_test_score_u.append((idx,array))

                        cv_train_scores_t[cvround] =  list_train_score_t
                        cv_train_scores_u[cvround] =  list_train_score_u
                        cv_test_scores_t[cvround] =  list_test_score_t
                        cv_test_scores_u[cvround] =  list_test_score_u
                    else:
                        list_train_score_t = []
                        for idx, array in cv_train_scores_t[cvround]:
                            array[currload] = array[currload]
                            list_train_score_t.append((idx,array))

                        list_train_score_u = []
                        for idx, array in cv_train_scores_u[cvround]:
                            array[currload] = array[currload]
                            list_train_score_u.append((idx,array))

                        list_test_score_t = []
                        for idx, array in cv_test_scores_t[cvround]:
                            array[currload] = array[currload]
                            list_test_score_t.append((idx,array))

                        list_test_score_u = []
                        for idx, array in cv_test_scores_u[cvround]:
                            array[currload] = array[currload]
                            list_test_score_u.append((idx,array))

                        cv_train_scores_t[cvround] =  list_train_score_t
                        cv_train_scores_u[cvround] =  list_train_score_u
                        cv_test_scores_t[cvround] =  list_test_score_t
                        cv_test_scores_u[cvround] =  list_test_score_u

            # Calculate Q-squareds
            q_squaredy = 1 - (pressy / ssy)
            q_squaredx = 1 - (pressx / ssx)

            # Store everything...
            self.cvParameters = {
                'PLS': {
                    'Q2X': q_squaredx, 'Q2Y': q_squaredy,
                    'MeanR2X_Training': numpy.mean(R2X_training), 'StdevR2X_Training': numpy.std(R2X_training),
                    'MeanR2Y_Training': numpy.mean(R2Y_training), 'StdevR2Y_Training': numpy.std(R2X_training),
                    'MeanR2X_Test': numpy.mean(R2X_test), 'StdevR2X_Test': numpy.std(R2X_test),
                    'MeanR2Y_Test': numpy.mean(R2Y_test), 'StdevR2Y_Test': numpy.std(R2Y_test),
                    }, 
                'DA': {}
                }
                       
            # Save everything found during CV
            if outputdist is True:
                self.cvParameters['PLS']['CVR2X_Training'] = R2X_training
                self.cvParameters['PLS']['CVR2Y_Training'] = R2Y_training
                self.cvParameters['PLS']['CVR2X_Test'] = R2X_test
                self.cvParameters['PLS']['CVR2Y_Test'] = R2Y_test
                self.cvParameters['PLS']['CV_Loadings_q'] = cv_loadings_q
                self.cvParameters['PLS']['CV_Loadings_p'] = cv_loadings_p
                self.cvParameters['PLS']['CV_Weights_c'] = cv_weights_c
                self.cvParameters['PLS']['CV_Weights_w'] = cv_weights_w
                self.cvParameters['PLS']['CV_Rotations_ws'] = cv_rotations_ws
                self.cvParameters['PLS']['CV_Rotations_cs'] = cv_rotations_cs
                self.cvParameters['PLS']['CV_TrainScores_t'] = cv_train_scores_t
                self.cvParameters['PLS']['CV_TrainScores_u'] = cv_train_scores_u
                self.cvParameters['PLS']['CV_TestScores_t'] = cv_test_scores_t
                self.cvParameters['PLS']['CV_TestScores_u'] = cv_test_scores_u
                self.cvParameters['PLS']['CV_Beta'] = cv_betacoefs
                self.cvParameters['PLS']['CV_VIPw'] = cv_vipsw
                # CV Train parameters - so we can keep a look on model performance in training set
                self.cvParameters['DA']['CV_TrainMCC'] = cv_trainmatthews_mcc
                self.cvParameters['DA']['CV_TrainRecall'] = cv_trainrecall
                self.cvParameters['DA']['CV_TrainPrecision'] = cv_trainprecision
                self.cvParameters['DA']['CV_TrainAccuracy'] = cv_trainaccuracy
                self.cvParameters['DA']['CV_TrainF1'] = cv_trainf1
                self.cvParameters['DA']['CV_Train0-1Loss'] = cv_trainzerooneloss
                self.cvParameters['DA']['CV_TrainConfusionMatrix'] = cv_trainconfusionmatrix
                self.cvParameters['DA']['CV_TrainSamplePrediction'] = cv_trainclasspredictions
                self.cvParameters['DA']['CV_TrainMisclassifiedsamples'] = cv_trainmisclassifiedsamples
                self.cvParameters['DA']['CV_TrainROC_fpr'] = cv_trainroc_fpr
                self.cvParameters['DA']['CV_TrainROC_tpr'] = cv_trainroc_tpr
                self.cvParameters['DA']['CV_TrainAUC'] = cv_testauc
                # CV Test set metrics - The metrics which matter to benchmark classifier
                self.cvParameters['DA']['CV_TestMCC'] = cv_testmatthews_mcc
                self.cvParameters['DA']['CV_TestRecall'] = cv_testrecall
                self.cvParameters['DA']['CV_TestPrecision'] = cv_testprecision
                self.cvParameters['DA']['CV_TestAccuracy'] = cv_testaccuracy
                self.cvParameters['DA']['CV_TestF1'] = cv_testf1
                self.cvParameters['DA']['CV_Test0-1Loss'] = cv_testzerooneloss
                self.cvParameters['DA']['CV_TestConfusionMatrix'] = cv_testconfusionmatrix
                self.cvParameters['DA']['CV_TestSamplePrediction'] = cv_testclasspredictions
                self.cvParameters['DA']['CV_TestMisclassifiedsamples'] = cv_testmisclassifiedsamples
                self.cvParameters['DA']['CV_TestROC_fpr'] = cv_testroc_fpr
                self.cvParameters['DA']['CV_TestROC_tpr'] = cv_testroc_tpr
                self.cvParameters['DA']['CV_TestAUC'] = cv_testauc
                
            return None

        except TypeError as terp:
            raise terp

    def _residual_ssx(self, x):
        """
        Residual Sum of Squares.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features]
            Data matrix.
        
        Returns
        -------
        RSSX : numpy.ndarray
            The residual Sum of Squares per sample.
        """
        pred_scores = self.transform(x)

        x_reconstructed = self.scaler.transform(self.inverse_transform(pred_scores))
        xscaled = self.scaler.transform(x)
        residuals = numpy.sum(numpy.square(xscaled - x_reconstructed), axis=1)
        return residuals

    def permutation_test(self, x, y, nperms=100, cv_method=model_selection.KFold(7, shuffle=True), outputdist=False, **permtest_kwargs):
        """
        Permutation test.

        Permutation test for the classifier and most model parameters.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.
        nperms : int
            Number of permutations.
        cv_method : BaseCrossValidator or BaseShuffleSplit
            An instance of a scikit-learn CrossValidator object.
        outputdist : bool
            Output the permutation test parameters.
        permtest_kwargs : kwargs
            Keyword arguments to be passed to the sklearn.Pipeline during cross-validation.

        Returns
        -------
        perm_params : dict
            Adds a dictionary permParameters to the object, containing the permutation test results.

        Raises
        ------
        ValueError
            If the x and y data matrix is invalid.
        """
        try:
            # Check if global model is fitted... and if not, fit it to provided x and y
            if self._isfitted is False or self.loadings_p is None:
                self.fit(x, y, **permtest_kwargs)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            permute_class = deepcopy(self)

            # Number of classes to select tell binary from multi-class discrimination parameter calculation
            n_classes = numpy.unique(y).size

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1

            # The y variable expected is a single vector with ints as class label - binary
            # and multiclass classification are allowed but not multilabel so this will work.
            # but for the PLS part in case of more than 2 classes a dummy matrix is constructed and kept separately
            # throughout
            if y.ndim == 1:
                # y = y.reshape(-1, 1)
                if self.n_classes > 2:
                    y_pls = pandas.get_dummies(y).values
                    y_nvars = y_pls.shape[1]
                else:
                    y_nvars = 1
                    y_pls = y
            else:
                raise TypeError('Please supply a dummy vector with integer as class membership')

            # Initialize data structures for permuted distributions
            perm_loadings_p = numpy.zeros((nperms, x_nvars, self.ncomps))
            perm_loadings_q = numpy.zeros((nperms, y_nvars, self.ncomps))
            perm_weights_w = numpy.zeros((nperms, x_nvars, self.ncomps))
            perm_weights_c = numpy.zeros((nperms, y_nvars, self.ncomps))
            perm_rotations_ws = numpy.zeros((nperms, x_nvars, self.ncomps))
            perm_rotations_cs = numpy.zeros((nperms, y_nvars, self.ncomps))
            perm_beta = numpy.zeros((nperms, y_nvars, x_nvars))
            perm_vipsw = numpy.zeros((nperms, y_nvars, x_nvars))

            perm_R2Y = numpy.zeros(nperms)
            perm_R2X = numpy.zeros(nperms)
            perm_Q2Y = numpy.zeros(nperms)
            perm_Q2X = numpy.zeros(nperms)

            perm_trainaccuracy = numpy.zeros(nperms)
            perm_trainprecision = numpy.zeros(nperms)
            perm_trainrecall = numpy.zeros(nperms)
            perm_trainf1 = numpy.zeros(nperms)
            perm_trainmatthews_mcc = numpy.zeros(nperms)
            perm_trainzerooneloss = numpy.zeros(nperms)
            perm_trainmisclassifiedsamples = list()
            
            perm_testaccuracy = numpy.zeros(nperms)
            perm_testprecision = numpy.zeros(nperms)
            perm_testrecall = numpy.zeros(nperms)
            perm_testf1 = numpy.zeros(nperms)
            perm_testmatthews_mcc = numpy.zeros(nperms)
            perm_testzerooneloss = numpy.zeros(nperms)
            perm_testmisclassifiedsamples = list()

            if self.n_classes == 2:
                perm_trainroc_fpr = numpy.zeros((nperms, len(y), 1))
                perm_trainroc_fpr[:] = numpy.nan
                perm_trainroc_tpr = numpy.zeros((nperms, len(y), 1))
                perm_trainroc_tpr[:] = numpy.nan
                perm_trainauc = numpy.zeros((nperms, 1))

                perm_testroc_fpr = numpy.zeros((nperms, len(y), 1))
                perm_testroc_fpr[:] = numpy.nan
                perm_testroc_tpr = numpy.zeros((nperms, len(y), 1))
                perm_testroc_tpr[:] = numpy.nan
                perm_testauc = numpy.zeros((nperms, 1))
            else:
                perm_trainroc_fpr = numpy.zeros((nperms, len(y), y_nvars))
                perm_trainroc_fpr[:] = numpy.nan
                perm_trainroc_tpr = numpy.zeros((nperms, len(y), y_nvars))
                perm_trainroc_tpr[:] = numpy.nan
                perm_trainauc = numpy.zeros((nperms, y_nvars))

                perm_testroc_fpr = numpy.zeros((nperms, len(y), y_nvars))
                perm_testroc_fpr[:] = numpy.nan
                perm_testroc_tpr = numpy.zeros((nperms, len(y), y_nvars))
                perm_testroc_tpr[:] = numpy.nan
                perm_testauc = numpy.zeros((nperms, y_nvars))

            for permutation in range(0, nperms):
                # Permute labels
                perm_y = numpy.random.permutation(y)

                # ... Fit model and replace original data
                permute_class.fit(x, perm_y, **permtest_kwargs)
                permute_class.cross_validation(x, perm_y, cv_method=cv_method, outputdist = True, **permtest_kwargs)

                # Predict y
                y_pred = permute_class.predict(x)

                # Store the loadings for each permutation component-wise
                perm_loadings_p[permutation, :, :] = permute_class.loadings_p
                perm_loadings_q[permutation, :, :] = permute_class.loadings_q
                perm_weights_w[permutation, :, :] = permute_class.weights_w
                perm_weights_c[permutation, :, :] = permute_class.weights_c
                perm_rotations_ws[permutation, :, :] = permute_class.rotations_ws
                perm_rotations_cs[permutation, :, :] = permute_class.rotations_cs
                perm_beta[permutation, :, :] = permute_class.beta_coeffs.T
                perm_vipsw[permutation, :, :] = permute_class.VIP().T

                perm_R2Y[permutation] = permute_class.modelParameters['PLS']['R2Y']
                perm_R2X[permutation] = permute_class.modelParameters['PLS']['R2X']
                perm_Q2Y[permutation] = permute_class.cvParameters['PLS']['Q2Y']
                perm_Q2X[permutation] = permute_class.cvParameters['PLS']['Q2X']

                # Train metrics
                perm_trainaccuracy[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainAccuracy'])
                perm_trainprecision[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainPrecision'])
                perm_trainrecall[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainRecall'])
                perm_trainf1[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainF1'])
                perm_trainmatthews_mcc[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainMCC'])
                perm_trainzerooneloss[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_Train0-1Loss'])
                train_misclassified_samples = perm_y[numpy.where(perm_y.ravel() != y_pred.ravel())[0]]
                perm_trainmisclassifiedsamples.append(train_misclassified_samples)

                # Test metrics
                perm_testaccuracy[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestAccuracy'])
                perm_testprecision[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestPrecision'])
                perm_testrecall[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestRecall'])
                perm_testf1[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestF1'])
                perm_testmatthews_mcc[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestMCC'])
                perm_testzerooneloss[permutation] = numpy.nanmean(permute_class.cvParameters['DA']['CV_Test0-1Loss'])
                test_misclassified_samples = perm_y[numpy.where(perm_y.ravel() != y_pred.ravel())[0]]
                perm_testmisclassifiedsamples.append(test_misclassified_samples)
            
                if n_classes == 2:
                    # ROC AUC
                    fpr_train_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainROC_fpr'], axis = 0)
                    tpr_train_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainROC_tpr'], axis = 0)
                    auc_train_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainAUC'], axis = 0)
                    
                    fpr_test_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestROC_fpr'], axis = 0)
                    tpr_test_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestROC_tpr'], axis = 0)
                    auc_test_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestAUC'], axis = 0)

                    perm_trainroc_fpr[permutation, :len(fpr_train_mean), 0] = fpr_train_mean
                    perm_trainroc_tpr[permutation, :len(tpr_train_mean), 0] = tpr_train_mean
                    perm_trainauc[permutation] = auc_train_mean

                    perm_testroc_fpr[permutation, :len(fpr_test_mean), 0] = fpr_test_mean
                    perm_testroc_tpr[permutation, :len(tpr_test_mean), 0] = tpr_test_mean
                    perm_testauc[permutation] = auc_test_mean
                else:
                    # ROC AUC
                    fpr_train_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainROC_fpr'], axis = 0)
                    tpr_train_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainROC_tpr'], axis = 0)
                    auc_train_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TrainAUC'], axis = 0)

                    fpr_test_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestROC_fpr'], axis = 0)
                    tpr_test_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestROC_tpr'], axis = 0)
                    auc_test_mean = numpy.nanmean(permute_class.cvParameters['DA']['CV_TestAUC'], axis = 0)

                    for predclass in range(permute_class.n_classes):
                        perm_trainroc_fpr[permutation, :len(fpr_train_mean[:, predclass]), predclass] = fpr_train_mean[:, predclass]
                        perm_trainroc_tpr[permutation, :len(tpr_train_mean[:, predclass]), predclass] = tpr_train_mean[:, predclass]
                        perm_trainauc[permutation, predclass] = auc_train_mean[predclass]

                        perm_testroc_fpr[permutation, :len(fpr_test_mean[:, predclass]), predclass] = fpr_test_mean[:, predclass]
                        perm_testroc_tpr[permutation, :len(tpr_test_mean[:, predclass]), predclass] = tpr_test_mean[:, predclass]
                        perm_testauc[permutation, predclass] = auc_test_mean[predclass]

            # Align model parameters due to sign indeterminacy.
            # Solution provided is to select the sign that gives a more similar profile to the
            # Loadings calculated with the whole data.
            for perm_round in range(0, nperms):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = numpy.argmin(numpy.array(
                        [numpy.sum(numpy.abs(self.loadings_p[:, currload] - perm_loadings_p[perm_round, :, currload])),
                         numpy.sum(numpy.abs(self.loadings_p[:, currload] - perm_loadings_p[perm_round, :, currload] * -1))]))
                    if choice == 1:
                        perm_loadings_p[perm_round, :, currload] = -1 * perm_loadings_p[perm_round, :, currload]
                        perm_loadings_q[perm_round, :, currload] = -1 * perm_loadings_q[perm_round, :, currload]
                        perm_weights_w[perm_round, :, currload] = -1 * perm_weights_w[perm_round, :, currload]
                        perm_weights_c[perm_round, :, currload] = -1 * perm_weights_c[perm_round, :, currload]
                        perm_rotations_ws[perm_round, :, currload] = -1 * perm_rotations_ws[perm_round, :, currload]
                        perm_rotations_cs[perm_round, :, currload] = -1 * perm_rotations_cs[perm_round, :, currload]

            # Pack everything into a dictionary data structure and return
            # Store everything...
            self.permutationParameters = {
                'PLS': {}, 
                'DA': {},
                'p-values': {},
                }
            self.permutationParameters['PLS']['R2Y'] = perm_R2Y
            self.permutationParameters['PLS']['R2X'] = perm_R2X
            self.permutationParameters['PLS']['Q2Y'] = perm_Q2Y
            self.permutationParameters['PLS']['Q2X'] = perm_Q2X
            self.permutationParameters['PLS']['Loadings_p'] = perm_loadings_p
            self.permutationParameters['PLS']['Loadings_q'] = perm_loadings_q
            self.permutationParameters['PLS']['Weights_c'] = perm_weights_c
            self.permutationParameters['PLS']['Weights_w'] = perm_weights_w
            self.permutationParameters['PLS']['Rotations_ws'] = perm_rotations_ws
            self.permutationParameters['PLS']['Rotations_cs'] = perm_rotations_cs
            self.permutationParameters['PLS']['Beta'] = perm_beta
            self.permutationParameters['PLS']['VIPw'] = perm_vipsw
            
            self.permutationParameters['DA']['Perm_TrainAccuracy'] = perm_trainaccuracy
            self.permutationParameters['DA']['Perm_TrainPrecision'] = perm_trainprecision
            self.permutationParameters['DA']['Perm_TrainRecall'] = perm_trainrecall
            self.permutationParameters['DA']['Perm_TrainMatthewsMCC'] = perm_trainmatthews_mcc
            self.permutationParameters['DA']['Perm_TrainF1'] = perm_trainf1
            self.permutationParameters['DA']['Perm_Train0-1Loss'] = perm_trainzerooneloss
            self.permutationParameters['DA']['Perm_TrainROC_fpr'] = perm_trainroc_fpr
            self.permutationParameters['DA']['Perm_TrainROC_tpr'] = perm_trainroc_tpr
            self.permutationParameters['DA']['Perm_TrainAUC'] = perm_trainauc
            self.permutationParameters['DA']['Perm_TrainMisclassifiedsamples'] = perm_trainmisclassifiedsamples

            self.permutationParameters['DA']['Perm_TestAccuracy'] = perm_testaccuracy
            self.permutationParameters['DA']['Perm_TestPrecision'] = perm_testprecision
            self.permutationParameters['DA']['Perm_TestRecall'] = perm_testrecall
            self.permutationParameters['DA']['Perm_TestMatthewsMCC'] = perm_testmatthews_mcc
            self.permutationParameters['DA']['Perm_TestF1'] = perm_testf1
            self.permutationParameters['DA']['Perm_Test0-1Loss'] = perm_testzerooneloss
            self.permutationParameters['DA']['Perm_TestROC_fpr'] = perm_testroc_fpr
            self.permutationParameters['DA']['Perm_TestROC_tpr'] = perm_testroc_tpr
            self.permutationParameters['DA']['Perm_TestAUC'] = perm_testauc
            self.permutationParameters['DA']['Perm_TestMisclassifiedsamples'] = perm_testmisclassifiedsamples

            # Calculate p-value for some of the metrics of interest
            obs_q2y = self.cvParameters['PLS']['Q2Y']
            obs_trainAUC = numpy.nanmean(self.cvParameters['DA']['CV_TrainAUC'], axis = 0)
            obs_testAUC = numpy.nanmean(self.cvParameters['DA']['CV_TestAUC'], axis = 0)
            obs_trainf1 = numpy.nanmean(self.cvParameters['DA']['CV_TrainF1'], axis = 0)
            obs_testf1 = numpy.nanmean(self.cvParameters['DA']['CV_TestF1'], axis = 0)

            self.permutationParameters['p-values']['Q2Y'] = (len(numpy.where(perm_Q2Y >= obs_q2y)[0]) + 1) / (nperms + 1)
            self.permutationParameters['p-values']['TrainAUC'] = (len(numpy.where(perm_trainauc >= obs_trainAUC)[0]) + 1) / (nperms + 1)
            self.permutationParameters['p-values']['TestAUC'] = (len(numpy.where(perm_testauc >= obs_testAUC)[0]) + 1) / (nperms + 1)
            self.permutationParameters['p-values']['Trainf1'] = (len(numpy.where(perm_trainf1 >= obs_trainf1)[0]) + 1) / (nperms + 1)
            self.permutationParameters['p-values']['Testf1'] = (len(numpy.where(perm_testf1 >= obs_testf1)[0]) + 1) / (nperms + 1)
            return

        except ValueError as exp:
            raise exp

    def bootstrap_test(self, x, y, nboots=100, stratify = True, outputdist=False, **bootstrap_kwargs):
        """
        Bootstrapping.

        Bootstrapping for confidence intervalls of the classifier and most model parameters.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_samples, n_features] or None
            X Data matrix in the original data space.
        y : numpy.ndarray, shape [n_samples, n_responses] or None
            Y Data matrix in the original data space.
        nboots : int
            Number of bootstraps.
        stratify : bool
            Use stratification.
        outputdist : bool
            Output the bootstrapping parameters.
        bootstrap_kwargs : kwargs
            Keyword arguments to be passed to the sklearn.Pipeline during cross-validation.

        Returns
        -------
        perm_params : dict
            Adds a dictionary permParameters to the object, containing the permutation test results.

        Raises
        ------
        ValueError
            If the x and y data matrix is invalid.
        """
        try:
            # Check if global model is fitted... and if not, fit it to provided x and y
            if self._isfitted is False or self.loadings_p is None:
                self.fit(x, y, **bootstrap_kwargs)
            # Make a copy of the object, to ensure the internal state doesn't come out differently from the
            # cross validation method call...
            bootstrap_pipeline = deepcopy(self)
            n_samples = int(0.7*len(y))
            seed = numpy.random.seed(None)
            bootidx = []
            bootidx_oob = []
            for i in range(nboots):
                #bootidx_i = numpy.random.choice(len(self.Y), len(self.Y))
                if stratify == True:
                    bootidx_i = sklearn.utils.resample(list(range(len(y))), stratify=y)
                else:
                    bootidx_i = sklearn.utils.resample(list(range(len(y))))
                bootidx_oob_i = numpy.array(list(set(range(len(y))) - set(bootidx_i)))
                bootidx.append(bootidx_i)
                bootidx_oob.append(bootidx_oob_i)

            # Number of classes to select tell binary from multi-class discrimination parameter calculation
            n_classes = numpy.unique(y).size

            if x.ndim > 1:
                x_nvars = x.shape[1]
            else:
                x_nvars = 1
            # The y variable expected is a single vector with ints as class label - binary
            # and multiclass classification are allowed but not multilabel so this will work.
            # but for the PLS part in case of more than 2 classes a dummy matrix is constructed and kept separately
            # throughout
            if y.ndim == 1:
                # y = y.reshape(-1, 1)
                if self.n_classes > 2:
                    y_pls = pandas.get_dummies(y).values
                    y_nvars = y_pls.shape[1]
                else:
                    y_pls = y
                    y_nvars = 1
            else:
                raise TypeError('Please supply a dummy vector with integer as class membership')


            # Initialize list structures to contain the fit
            boot_loadings_p = numpy.zeros((nboots, x_nvars, self.ncomps))
            boot_loadings_q = numpy.zeros((nboots, y_nvars, self.ncomps))
            boot_weights_w = numpy.zeros((nboots, x_nvars, self.ncomps))
            boot_weights_c = numpy.zeros((nboots, y_nvars, self.ncomps))
            boot_train_scores_t = list()
            boot_train_scores_u = list()

            # CV test scores more informative for ShuffleSplit than KFold but kept here anyway
            boot_test_scores_t = list()
            boot_test_scores_u = list()

            boot_rotations_ws = numpy.zeros((nboots, x_nvars, self.ncomps))
            boot_rotations_cs = numpy.zeros((nboots, y_nvars, self.ncomps))
            boot_betacoefs = numpy.zeros((nboots, y_nvars, x_nvars))
            boot_vipsw = numpy.zeros((nboots, y_nvars, x_nvars))

            boot_trainprecision = numpy.zeros(nboots)
            boot_trainrecall = numpy.zeros(nboots)
            boot_trainaccuracy = numpy.zeros(nboots)
            boot_trainmatthews_mcc = numpy.zeros(nboots)
            boot_trainzerooneloss = numpy.zeros(nboots)
            boot_trainf1 = numpy.zeros(nboots)
            boot_trainclasspredictions = list()
            boot_trainconfusionmatrix = list()
            boot_trainmisclassifiedsamples = list()

            boot_testprecision = numpy.zeros(nboots)
            boot_testrecall = numpy.zeros(nboots)
            boot_testaccuracy = numpy.zeros(nboots)
            boot_testmatthews_mcc = numpy.zeros(nboots)
            boot_testzerooneloss = numpy.zeros(nboots)
            boot_testf1 = numpy.zeros(nboots)
            boot_testclasspredictions = list()
            boot_testconfusionmatrix = list()
            boot_testmisclassifiedsamples = list()

            if self.n_classes == 2:
                boot_trainroc_fpr = numpy.zeros((nboots, len(y), 1))
                boot_trainroc_fpr[:] = numpy.nan
                boot_trainroc_tpr = numpy.zeros((nboots, len(y), 1))
                boot_trainroc_tpr[:] = numpy.nan
                boot_trainauc = numpy.zeros((nboots, 1))

                boot_testroc_fpr = numpy.zeros((nboots, len(y), 1))
                boot_testroc_fpr[:] = numpy.nan
                boot_testroc_tpr = numpy.zeros((nboots, len(y), 1))
                boot_testroc_tpr[:] = numpy.nan
                boot_testauc = numpy.zeros((nboots, 1))
            else:
                boot_trainroc_fpr = numpy.zeros((nboots, len(y), y_nvars))
                boot_trainroc_fpr[:] = numpy.nan
                boot_trainroc_tpr = numpy.zeros((nboots, len(y), y_nvars))
                boot_trainroc_tpr[:] = numpy.nan
                boot_trainauc = numpy.zeros((nboots, y_nvars))

                boot_testroc_fpr = numpy.zeros((nboots, len(y), y_nvars))
                boot_testroc_fpr[:] = numpy.nan
                boot_testroc_tpr = numpy.zeros((nboots, len(y), y_nvars))
                boot_testroc_tpr[:] = numpy.nan
                boot_testauc = numpy.zeros((nboots, y_nvars))
    
            # Initialise predictive residual sum of squares variable (for whole CV routine)
            pressy = 0
            pressx = 0

            # Calculate Sum of Squares SS in whole dataset for future calculations
            ssx = numpy.sum(numpy.square(bootstrap_pipeline.x_scaler.fit_transform(x)))
            ssy = numpy.sum(numpy.square(bootstrap_pipeline._y_scaler.fit_transform(y_pls.reshape(-1, 1))))

            # As assessed in the test set..., opposed to PRESS
            R2X_training = numpy.zeros(nboots)
            R2Y_training = numpy.zeros(nboots)
            # R2X and R2Y assessed in the test set
            R2X_test = numpy.zeros(nboots)
            R2Y_test = numpy.zeros(nboots)

            # Cross validation
            for boot_i, idx_i, idx_oob_i in zip(range(nboots), bootidx, bootidx_oob):
            
                # split the data explicitly
                train = idx_i
                test = idx_oob_i

                # Check dimensions for the indexing
                ytrain = y[train]
                ytest = y[test]
                if x_nvars == 1:
                    xtrain = x[train]
                    xtest = x[test]
                else:
                    xtrain = x[train, :]
                    xtest = x[test, :]
     

                bootstrap_pipeline.fit(xtrain, ytrain, **bootstrap_kwargs)

                # Prepare the scaled X and Y test data

                # Comply with the sklearn scaler behaviour
                if xtest.ndim == 1:
                    xtest = xtest.reshape(-1, 1)
                    xtrain = xtrain.reshape(-1, 1)
                # Fit the training data

                xtest_scaled = bootstrap_pipeline.x_scaler.transform(xtest)

                R2X_training[boot_i] = ChemometricsPLS.score(bootstrap_pipeline, xtrain, ytrain, 'x')
                R2Y_training[boot_i] = ChemometricsPLS.score(bootstrap_pipeline, xtrain, ytrain, 'y')
                
                if y_pls.ndim > 1:
                    yplstest = y_pls[test, :]

                else:
                    yplstest = y_pls[test].reshape(-1, 1)

                # Use super here  for Q2
                ypred = ChemometricsPLS.predict(bootstrap_pipeline, x=xtest, y=None)
                xpred = ChemometricsPLS.predict(bootstrap_pipeline, x=None, y=ytest)

                xpred = bootstrap_pipeline.x_scaler.transform(xpred).squeeze()
                ypred = bootstrap_pipeline._y_scaler.transform(ypred).squeeze()

                curr_pressx = numpy.sum(numpy.square(xtest_scaled - xpred))
                curr_pressy = numpy.sum(numpy.square(bootstrap_pipeline._y_scaler.transform(yplstest).squeeze() - ypred))

                R2X_test[boot_i] = ChemometricsPLS.score(bootstrap_pipeline, xtest, yplstest, 'x')
                R2Y_test[boot_i] = ChemometricsPLS.score(bootstrap_pipeline, xtest, yplstest, 'y')

                pressx += curr_pressx
                pressy += curr_pressy

                boot_loadings_p[boot_i, :, :] = bootstrap_pipeline.loadings_p
                boot_loadings_q[boot_i, :, :] = bootstrap_pipeline.loadings_q
                boot_weights_w[boot_i, :, :] = bootstrap_pipeline.weights_w
                boot_weights_c[boot_i, :, :] = bootstrap_pipeline.weights_c
                boot_rotations_ws[boot_i, :, :] = bootstrap_pipeline.rotations_ws
                boot_rotations_cs[boot_i, :, :] = bootstrap_pipeline.rotations_cs
                boot_betacoefs[boot_i, :, :] = bootstrap_pipeline.beta_coeffs.T
                boot_vipsw[boot_i, :, :] = bootstrap_pipeline.VIP().T

                # Training metrics
                boot_trainaccuracy[boot_i] = bootstrap_pipeline.modelParameters['DA']['Accuracy']
                boot_trainprecision[boot_i] = bootstrap_pipeline.modelParameters['DA']['Precision']
                boot_trainrecall[boot_i] = bootstrap_pipeline.modelParameters['DA']['Recall']
                boot_trainf1[boot_i] = bootstrap_pipeline.modelParameters['DA']['F1']
                boot_trainmatthews_mcc[boot_i] = bootstrap_pipeline.modelParameters['DA']['MatthewsMCC']
                boot_trainzerooneloss[boot_i] = bootstrap_pipeline.modelParameters['DA']['0-1Loss']

                # Check this indexes, same as CV scores

                boot_trainclasspredictions.append(
                    [*zip(train, bootstrap_pipeline.modelParameters['DA']['ClassPredictions'])])

                # Predict y
                y_pred = bootstrap_pipeline.predict(xtest)
                # Obtain the class score
                class_score = ChemometricsPLS.predict(bootstrap_pipeline, xtest)

                # DA parameter
                if n_classes == 2:
                    boot_testaccuracy[boot_i] = metrics.accuracy_score(ytest, y_pred)
                    boot_testprecision[boot_i] = metrics.precision_score(ytest, y_pred)
                    boot_testrecall[boot_i] = metrics.recall_score(ytest, y_pred)
                    boot_testf1[boot_i]  = metrics.f1_score(ytest, y_pred)
                    boot_testzerooneloss[boot_i] = metrics.zero_one_loss(ytest, y_pred)
                    boot_testmatthews_mcc[boot_i] = metrics.matthews_corrcoef(ytest, y_pred)

                    # ROC AUC
                    fpr_mean = bootstrap_pipeline.modelParameters['DA']['ROC_fpr']
                    tpr_mean = bootstrap_pipeline.modelParameters['DA']['ROC_tpr']
                    auc_mean = bootstrap_pipeline.modelParameters['DA']['AUC']

                    boot_trainroc_fpr[boot_i, :len(fpr_mean), 0] = fpr_mean[:,0]
                    boot_trainroc_tpr[boot_i, :len(tpr_mean), 0] = tpr_mean[:,0]
                    boot_trainauc[boot_i] = auc_mean[0]

                    fpr, tpr, _ = metrics.roc_curve(ytest, class_score.ravel(), drop_intermediate = False)
                    boot_testroc_fpr[boot_i, :len(fpr), 0] = fpr
                    boot_testroc_tpr[boot_i, :len(tpr), 0] = tpr
                    boot_testauc[boot_i] = metrics.auc(fpr, tpr)

                else:
                    boot_testaccuracy[boot_i] = metrics.accuracy_score(ytest, y_pred)
                    boot_testprecision[boot_i] = metrics.precision_score(ytest, y_pred, average='weighted')
                    boot_testrecall[boot_i] = metrics.recall_score(ytest, y_pred, average='weighted')
                    boot_testf1[boot_i] = metrics.f1_score(ytest, y_pred, average='weighted')
                    boot_testzerooneloss[boot_i] = metrics.zero_one_loss(ytest, y_pred)
                    boot_testmatthews_mcc[boot_i] = numpy.nan
                    
                    # ROC AUC
                    fpr_mean = bootstrap_pipeline.modelParameters['DA']['ROC_fpr']
                    tpr_mean = bootstrap_pipeline.modelParameters['DA']['ROC_tpr']
                    auc_mean = bootstrap_pipeline.modelParameters['DA']['AUC']

                    # Generate multiple ROC curves - one for each class the multiple class case
                    for predclass in range(bootstrap_pipeline.n_classes):
                        # ROC
                        boot_trainroc_fpr[boot_i, :len(fpr_mean[:, predclass]), predclass] = fpr_mean[:, predclass]
                        boot_trainroc_tpr[boot_i, :len(tpr_mean[:, predclass]), predclass] = tpr_mean[:, predclass]
                        boot_trainauc[boot_i, predclass] = auc_mean[predclass]

                        fpr, tpr, _ = metrics.roc_curve(ytest, class_score[:, predclass], pos_label=predclass, drop_intermediate = False)
                        boot_testroc_fpr[boot_i, :len(fpr), predclass] = fpr
                        boot_testroc_tpr[boot_i, :len(tpr), predclass] = tpr
                        boot_testauc[boot_i, predclass] = metrics.auc(fpr, tpr)

                # TODO check the roc curve in train and test set
                # Check the actual indexes in the original samples
                test_misclassified_samples = test[numpy.where(ytest.ravel() != y_pred.ravel())[0]]
                test_classpredictions = [*zip(test, y_pred)]
                test_conf_matrix = metrics.confusion_matrix(ytest, y_pred)

                # Check this indexes, same as CV scores
                boot_testmisclassifiedsamples.append(test_misclassified_samples)
                boot_testconfusionmatrix.append(test_conf_matrix)
                boot_testclasspredictions.append(test_classpredictions)

            # Do a proper investigation on how to get CV scores decently
            # Align model parameters to account for sign indeterminacy.
            # The criteria here used is to select the sign that gives a more similar profile (by L1 distance) to the loadings from
            # on the model fitted with the whole data. Any other parameter can be used, but since the loadings in X capture
            # the covariance structure in the X data block, in theory they should have more pronounced features even in cases of
            # null X-Y association, making the sign flip more resilient.
            for boot_i in range(0, nboots):
                for currload in range(0, self.ncomps):
                    # evaluate based on loadings _p
                    choice = numpy.argmin(
                        numpy.array([numpy.sum(numpy.abs(self.loadings_p[:, currload] - boot_loadings_p[boot_i, :, currload])),
                                  numpy.sum(numpy.abs(
                                      self.loadings_p[:, currload] - boot_loadings_p[boot_i, :, currload] * -1))]))
                    if choice == 1:
                        boot_loadings_p[boot_i, :, currload] = -1 * boot_loadings_p[boot_i, :, currload]
                        boot_loadings_q[boot_i, :, currload] = -1 * boot_loadings_q[boot_i, :, currload]
                        boot_weights_w[boot_i, :, currload] = -1 * boot_weights_w[boot_i, :, currload]
                        boot_weights_c[boot_i, :, currload] = -1 * boot_weights_c[boot_i, :, currload]
                        boot_rotations_ws[boot_i, :, currload] = -1 * boot_rotations_ws[boot_i, :, currload]
                        boot_rotations_cs[boot_i, :, currload] = -1 * boot_rotations_cs[boot_i, :, currload]
                        boot_train_scores_t.append([*zip(train, -1 * bootstrap_pipeline.scores_t)])
                        boot_train_scores_u.append([*zip(train, -1 * bootstrap_pipeline.scores_u)])
                        boot_test_scores_t.append([*zip(test, -1 * bootstrap_pipeline.scores_t)])
                        boot_test_scores_u.append([*zip(test, -1 * bootstrap_pipeline.scores_u)])
                    else:
                        boot_train_scores_t.append([*zip(train, bootstrap_pipeline.scores_t)])
                        boot_train_scores_u.append([*zip(train, bootstrap_pipeline.scores_u)])
                        boot_test_scores_t.append([*zip(test, bootstrap_pipeline.scores_t)])
                        boot_test_scores_u.append([*zip(test, bootstrap_pipeline.scores_u)])

            # Calculate Q-squareds
            q_squaredy = 1 - (pressy / ssy)
            q_squaredx = 1 - (pressx / ssx)

            # Store everything...
            self.bootstrapParameters = {
                'PLS': {
                    'Q2X': q_squaredx, 'Q2Y': q_squaredy,
                    'MeanR2X_Training': numpy.mean(R2X_training), 'StdevR2X_Training': numpy.std(R2X_training),
                    'MeanR2Y_Training': numpy.mean(R2Y_training), 'StdevR2Y_Training': numpy.std(R2X_training),
                    'MeanR2X_Test': numpy.mean(R2X_test), 'StdevR2X_Test': numpy.std(R2X_test),
                    'MeanR2Y_Test': numpy.mean(R2Y_test), 'StdevR2Y_Test': numpy.std(R2Y_test),
                    }, 
                'DA': {}
                }

            # Save everything found during CV
            if outputdist is True:
                self.bootstrapParameters['PLS']['bootR2X_Training'] = R2X_training
                self.bootstrapParameters['PLS']['bootR2Y_Training'] = R2Y_training
                self.bootstrapParameters['PLS']['bootR2X_Test'] = R2X_test
                self.bootstrapParameters['PLS']['bootR2Y_Test'] = R2Y_test
                self.bootstrapParameters['PLS']['boot_Loadings_q'] = boot_loadings_q
                self.bootstrapParameters['PLS']['boot_Loadings_p'] = boot_loadings_p
                self.bootstrapParameters['PLS']['boot_Weights_c'] = boot_weights_c
                self.bootstrapParameters['PLS']['boot_Weights_w'] = boot_weights_w
                self.bootstrapParameters['PLS']['boot_Rotations_ws'] = boot_rotations_ws
                self.bootstrapParameters['PLS']['boot_Rotations_cs'] = boot_rotations_cs
                self.bootstrapParameters['PLS']['boot_TestScores_t'] = boot_test_scores_t
                self.bootstrapParameters['PLS']['boot_TestScores_u'] = boot_test_scores_u
                self.bootstrapParameters['PLS']['boot_TrainScores_t'] = boot_train_scores_t
                self.bootstrapParameters['PLS']['boot_TrainScores_u'] = boot_train_scores_u
                self.bootstrapParameters['PLS']['boot_Beta'] = boot_betacoefs
                self.bootstrapParameters['PLS']['boot_VIPw'] = boot_vipsw
                # CV Train parameters - so we can keep a look on model performance in training set
                self.bootstrapParameters['DA']['boot_TrainMCC'] = boot_trainmatthews_mcc
                self.bootstrapParameters['DA']['boot_TrainRecall'] = boot_trainrecall
                self.bootstrapParameters['DA']['boot_TrainPrecision'] = boot_trainprecision
                self.bootstrapParameters['DA']['boot_TrainAccuracy'] = boot_trainaccuracy
                self.bootstrapParameters['DA']['boot_TrainF1'] = boot_trainf1
                self.bootstrapParameters['DA']['boot_Train0-1Loss'] = boot_trainzerooneloss
                self.bootstrapParameters['DA']['boot_TrainConfusionMatrix'] = boot_trainconfusionmatrix
                self.bootstrapParameters['DA']['boot_TrainSamplePrediction'] = boot_trainclasspredictions
                self.bootstrapParameters['DA']['boot_TrainMisclassifiedsamples'] = boot_trainmisclassifiedsamples
                self.bootstrapParameters['DA']['boot_TrainROC_fpr'] = boot_trainroc_fpr
                self.bootstrapParameters['DA']['boot_TrainROC_tpr'] = boot_trainroc_tpr
                self.bootstrapParameters['DA']['boot_TrainAUC'] = boot_trainauc
                # CV Test set metrics - The metrics which matter to benchmark classifier
                self.bootstrapParameters['DA']['boot_TestMCC'] = boot_testmatthews_mcc
                self.bootstrapParameters['DA']['boot_TestRecall'] = boot_testrecall
                self.bootstrapParameters['DA']['boot_TestPrecision'] = boot_testprecision
                self.bootstrapParameters['DA']['boot_TestAccuracy'] = boot_testaccuracy
                self.bootstrapParameters['DA']['boot_TestF1'] = boot_testf1
                self.bootstrapParameters['DA']['boot_Test0-1Loss'] = boot_testzerooneloss
                self.bootstrapParameters['DA']['boot_TestConfusionMatrix'] = boot_testconfusionmatrix
                self.bootstrapParameters['DA']['boot_TestSamplePrediction'] = boot_testclasspredictions
                self.bootstrapParameters['DA']['boot_TestMisclassifiedsamples'] = boot_testmisclassifiedsamples
                self.bootstrapParameters['DA']['boot_TestROC_fpr'] = boot_testroc_fpr
                self.bootstrapParameters['DA']['boot_TestROC_tpr'] = boot_testroc_tpr
                self.bootstrapParameters['DA']['boot_TestAUC'] = boot_testauc

            return None

        except ValueError as exp:
            raise exp

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

####################### Functions ######################

def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """
    Set scales of near constant features to 1.
    
    The goal is to avoid division by very small or zero values.
    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.
    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.

    Parameters
    ----------
    scale : array
        Scale to be corrected.
    copy : bool
        Create copy.
    constant_mask : array
        Masking array.

    Returns
    -------
    scale : array
        Corrected scale.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if numpy.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, numpy.ndarray):
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to suprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * numpy.finfo(scale.dtype).eps
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[constant_mask] = 1.0
        return scale

def scan_preprocessing(path_file, path_report, pre_signal_noise = 5, pre_precision = 20, pre_list_relevant = None, pre_labelsize_identification = 12, pre_figsize_identification = (4,4), pre_figsize_quantification = (4,4)):
    """
    Evaluate global scans.

    Main function of DSFIApy scan to evaluate global scan data generated by dilute-and-shoot flow-injection-analysis tandem mass spectrometry.

    Parameters
    ----------
    path_file : str
        Raw path string leading to DS-FIA-Data.
    path_report : str
        Raw path string leading to the corresponding development report, generated by dsfiapy method development.
    pre_signal_noise : float
        Signal-to-noise ratio threshold.
    pre_precision : float
        Precision relative standard deviation threshold.
    pre_list_relevant : list or None
        List of interesting metabolites for separate plots.
    pre_labelsize_identification : int
        Labelsize for initialization plots.
    pre_figsize_identification : tuple
        Figsize for initialization plots.
    pre_figsize_quantification : tuple
        Figsize for initialization plots.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Parse input
    inp = scan_preprocessing_parsing(path_file, path_report, pre_signal_noise, pre_precision, pre_list_relevant, pre_labelsize_identification, pre_figsize_identification, pre_figsize_quantification)
    # Create folder
    inp = scan_preprocessing_folder(inp)
    # Preprocess data
    inp = scan_preprocessing_main(inp)
    # Identification
    inp = scan_preprocessing_identification(inp)
    return inp

def scan_preprocessing_parsing(path_file, path_report, pre_signal_noise, pre_precision, pre_list_relevant, pre_labelsize_identification, pre_figsize_identification, pre_figsize_quantification):
    """
    Initialize preprocessing.

    Initialization function for preprocessing.

    Parameters
    ----------
    path_file : str
        Raw path string leading to DS-FIA-Data.
    path_report : str
        Raw path string leading to the corresponding development report, generated by dsfiapy method development.
    pre_signal_noise : float
        Signal-to-noise ratio threshold.
    pre_precision : float
        Precision relative standard deviation threshold.
    pre_list_relevant : list or None
        List of interesting metabolites for separate plots.
    pre_labelsize_identification : int
        Labelsize for initialization plots.
    pre_figsize_identification : tuple
        Figsize for initialization plots.
    pre_figsize_quantification : tuple
        Figsize for initialization plots.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    inp = {}
    # Paths
    inp['path_file_data'] = Path(path_file)
    inp['path_file_report'] = Path(path_report)
    # Parameter
    inp['pre_signal_noise'] = pre_signal_noise
    inp['pre_rsd'] = pre_precision
    inp['pre_list_relevant'] = pre_list_relevant
    # Plots
    inp['pre_labelsize'] = pre_labelsize_identification
    inp['pre_figsize_identification'] = pre_figsize_identification
    inp['pre_figsize_quantification'] = pre_figsize_quantification
    # Read data file
    inp['data_raw'] = pandas.read_excel(inp['path_file_data'])
    # Read report file of method development
    inp['information_report'] = pandas.read_excel(inp['path_file_report'], sheet_name = None)
    return inp

def scan_preprocessing_folder(inp):
    """
    Create folder.

    Create identification folder.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Create folder
    inp['path_folder'] = inp['path_file_data'].parent
    inp['path_evaluation'] = create_folder(inp['path_folder'], f'{inp["path_file_data"].stem}_scan')
    inp['path_evaluation_identification'] = create_folder(inp['path_evaluation'], '01_identification')
    return inp

def scan_preprocessing_main(inp):
    """
    Preprocess scan data.

    Preprocess raw data, filter signals, add references and cluster data.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Prepare dataframe
    inp = scan_preprocessing_preparation(inp)
    inp = scan_preprocessing_create_mapper_group(inp)
    # LOESS
    inp = scan_preprocessing_loess(inp)
    # Outlier detection
    inp = scan_preprocessing_outlier(inp)
    # Get thresholds
    df_filter = scan_preprocessing_filter(inp)
    # Map information
    inp = scan_preprocessing_create_mapper_report(inp)
    df_mapped = scan_preprocessing_reference(df_filter, inp).copy()
    # Cluster data
    inp = scan_preprocessing_cluster(df_mapped, inp)
    # Plot basic data
    scan_preprocessing_plotting(inp, variables = ['reference', 'mode', 'isobarics'], filename = 'cluster')
    scan_preprocessing_plotting(inp, variables = ['identification', 'sets'], filename = 'identification')
    # Apply quantification filter
    inp = scan_preprocessing_filter_quantitative(inp)
    return inp

def scan_preprocessing_preparation(inp):
    """
    Prepare scan data.

    Format data and filter relevant information.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Prepare data
    df_new = inp['data_raw'].copy()
    hold = df_new['Sample Name'].str.split('_', expand = True)
    # Read sample information
    df_new['Group'] = hold[0]
    df_new['Reactor'] = hold[1]
    df_new['Sample Number'] = hold[2].str.replace('[^0-9]', '', regex = True)
    df_new['Dilution factor'] = hold[3].str.replace('[^0-9]', '', regex = True)
    df_new['Batch'] = hold[4]
    # Order for LOESS
    df_new['Order'] = df_new['Sample ID']
    # Remove non-used peaks of Multiquant
    df_new = df_new[df_new['Used']==True].copy()
    # Get classes
    df_new = scan_preprocessing_preparation_get_classes(df_new)
    # Rename QC for batch
    df_new = scan_preprocessing_preparation_qc(df_new)
    # Filter columns
    df_new = df_new.filter(['Group','Reactor','Sample Number','Dilution factor','Batch','Order','Class','Component Name','Area','Signal / Noise']).copy()
    df_new['Area'] = df_new['Area'].apply(lambda x: float(x))
    df_new['Signal / Noise'] = df_new['Signal / Noise'].apply(lambda x: float(x))
    # Allocate dataframe
    inp['data_processed_raw'] = df_new.copy()
    # Save dataframe
    save_df(df_new, inp['path_evaluation_identification'], f'01_raw_SN{inp["pre_signal_noise"]}', index = False)
    return inp
    
def scan_preprocessing_preparation_get_classes(df):
    """
    Get sample classes.

    Get sample classes of parsed data.

    Parameters
    ----------
    df : dataframe
        Dataframe with Group category including class of samples.

    Returns
    -------
    df : dataframe
        Dataframe with Class category.
    """
    df.loc[df['Group'].str.contains("QC"),'Class'] = 'QC'
    df.loc[df['Group'].str.contains("Blank"),'Class'] = 'Blank'
    df['Class'] = df['Class'].fillna('Sample')
    return df

def scan_preprocessing_preparation_qc(df):
    """
    Set quality control per batch.

    Connect batch with quality control samples for quality control.

    Parameters
    ----------
    df : dataframe
        Dataframe with Class category.

    Returns
    -------
    df : dataframe
        Dataframe with quality control batches.
    """
    if 'QC' in set(df['Class']):
        df_qc = df[df['Class'] == 'QC'].copy()
        df_qc['Group'] = df_qc['Class']+' '+df_qc['Batch']
        df = df[df['Class']!='QC'].copy()
        df = df.append(df_qc)
    return df

def scan_preprocessing_loess(inp):
    """
    Loess correction.

    Calculate loess correction with value imputation and cross validation.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Allocation
    df = inp['data_processed_raw'].copy()
    
    # Apply LOESS
    print('LOESS correction')
    df_new = pandas.DataFrame()
    if 'QC' in set(df['Class']):
        for batch in df['Batch'].unique():
            df_batch = df[df['Batch'] == batch].copy().sort_values(['Order'])
            # Select random samples for plotting
            number_random_samples = 1
            random_samples = [(i,j) for i,j in enumerate(random.sample(set(df['Component Name']), number_random_samples))]

            # TODO: Blank correction        
            for comp in df_batch['Component Name'].unique():
                df_comp = df_batch[df_batch['Component Name'] == comp].copy()
                df_comp['Order_LOESS'] = range(1,len(df_comp)+1)
                # QC LOESS
                df_comp_qc = df_comp[df_comp['Class'] == 'QC'].copy()
                df_qc = scan_preprocessing_loess_impute(df_comp_qc)
                try:
                    x_qc = numpy.array(df_qc['Order_LOESS'])
                    y_qc = numpy.array(df_qc['Area'])
                    median_qc = numpy.nanmedian(df_comp_qc['Area'])
                    # LOWESS
                    y_qc_hat = scan_preprocessing_loess_cv(x_qc, y_qc)
                    # Interpolation
                    f_interpolated = scipy.interpolate.CubicSpline(x_qc, y_qc_hat)
                    # Save data for control charts
                    df_before = df_comp.copy()
                    # Apply LOESS, level to median of QC
                    df_comp['Area'] = df_comp['Area'] - f_interpolated(df_comp['Order_LOESS']) + median_qc
                    # Apply median normalization
                    df_comp['Area'] = df_comp['Area']/median_qc
                    # Append
                    df_new = df_new.append(df_comp)
                    # Plotting
                    if comp in [item[1] for item in random_samples]:
                        fig, axes = plt.subplots(ncols = 2, figsize = (6, 4), sharex = True, sharey = True)
                        random_samples_select = [item for item in random_samples if item[1] == comp][0]
                        for group in df_before['Group'].unique():
                            df_before_group = df_before[df_before['Group']==group].copy()
                            df_after_group = df_comp[df_comp['Group']==group].copy()
                            if 'QC ' in group:
                                size_dot = 10
                                color = 'red'
                                x_plot = numpy.linspace(numpy.nanmin(df_comp['Order_LOESS']), numpy.nanmax(df_comp['Order_LOESS']), 100)
                                y_plot = f_interpolated(x_plot)
                                axes[0].plot(x_plot, y_plot, linewidth = 1, color = color)
                            else:
                                size_dot = 6
                                color = 'blue'
                            axes[0].scatter(x = df_before_group['Order_LOESS'], y = df_before_group['Area'], s = size_dot, color = color)
                            axes[1].scatter(x = df_after_group['Order_LOESS'], y = df_after_group['Area'], s = size_dot, color = color)
                        axes[0].set_ylabel('Peak area   [cps min]', fontsize = 12, fontweight = 'bold')
                        ax_label = axes[1].twinx()
                        ax_label.tick_params(right = False, labelright = False)
                        ax_label.set_ylabel(comp, fontsize = 12, fontweight = 'bold')
                        
                        # Finalize plot
                        for ax in axes.flatten():
                            ax.tick_params(labelsize = 12)
                            ax.set_xlabel(r'Injection order $\to$', fontsize = 12, fontweight = 'bold')
                        fig.savefig(inp['path_evaluation_identification'].joinpath(f'LOESS_{comp}_SN{inp["pre_signal_noise"]}_{batch}.png'), bbox_inches = 'tight', dpi = 800)
                        fig.savefig(inp['path_evaluation_identification'].joinpath(f'LOESS_{comp}_SN{inp["pre_signal_noise"]}_{batch}.svg'), bbox_inches = 'tight', format = 'svg')
                        plt.close(fig)
                except:
                    df_new = df_new.append(df_comp)
    else:
        df_new = df.copy()
    inp['data_processed'] = df_new.copy()
    # Save dataframe
    save_df(df_new, inp['path_evaluation_identification'], f'02_loess_SN{inp["pre_signal_noise"]}', index = False)
    return inp

def scan_preprocessing_loess_cv(x,y):
    """
    LOESS cross validation.

    Cross validation procedure for LOESS correction smoothing factor.
    CV: Leave-one-out

    Parameters
    ----------
    x : array
        Sample position.
    y : array
        Response.

    Returns
    -------
    lowess_final : array
        LOESS corrected array.
    """
    # TODO: Check fraction list
    if len(x) <= 12:
        list_fractions = [2/3]
    else:
        list_fractions = [i / len(x) for i in range(4, len(x) + 1)]
    # Set iterations
    iterations = 3
    rmse = numpy.inf
    best_fraction = 1
    for fraction in list_fractions:
        rmse_temp = 0
        for idx_cv in x[1:-1]:
            idx_train = numpy.where(x != idx_cv)
            idx_test = numpy.where(x == idx_cv)
            x_cv = x[idx_train]
            y_cv = y[idx_train]
            y_cv_hat = scan_preprocessing_loess_ag(x_cv, y_cv, f = fraction, iter = iterations)
            f_interpolated = scipy.interpolate.CubicSpline(x_cv, y_cv_hat)
            rmse_temp = rmse_temp + numpy.sqrt((y[idx_test]-f_interpolated(x[idx_test]))**2)
        if rmse > rmse_temp:
            best_fraction = fraction
            rmse = rmse_temp
    fraction = best_fraction
    try:
        lowess_final = scan_preprocessing_loess_ag(x, y, f = fraction, iter = iterations)
    except:
        lowess_final = y
    return lowess_final

def scan_preprocessing_loess_ag(x, y, f=2. / 3., iter=3):
    """
    LOESS smoother.

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.

    Parameters
    ----------
    x : array
        Sample position.
    y : array
        Response.
    f : float
        Smoothing span.
    iter : int
        Iterations.

    Returns
    -------
    yest : array
        Response estimate.
    """
    n = len(x)
    r = int(numpy.ceil(f * n))
    h = [numpy.sort(numpy.abs(x - x[i]))[r] for i in range(n)]
    w = numpy.clip(numpy.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = numpy.zeros(n)
    delta = numpy.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = numpy.array([numpy.sum(weights * y), numpy.sum(weights * y * x)])
            A = numpy.array([[numpy.sum(weights), numpy.sum(weights * x)],
                          [numpy.sum(weights * x), numpy.sum(weights * x * x)]])
            beta = scipy.linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = numpy.median(numpy.abs(residuals))
        delta = numpy.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

def scan_preprocessing_loess_impute(df):
    """
    LOESS imputer.

    Value imputer for LOESS procedure.

    Parameters
    ----------
    df : dataframe
        Dataframe with potentially missing responses.

    Returns
    -------
    df : dataframe
        Dataframe with no missing responses.
    """
    # Mean imputation for LOESS, not raw data
    # Get series
    series_area = df['Area'].copy()
    # Get missing data for MAR imputation
    series_area[pandas.isna(series_area)] = numpy.nanmean(series_area)
    # If all missing, fill nan with 0, will be dropped later
    if True in set(pandas.isna(series_area)):
        series_area = series_area.fillna(0)
    df['Area'] = series_area.copy()
    return df

def scan_preprocessing_outlier(inp):
    """
    LOESS outlier handling.

    Outlier detection and removing for LOESS procedure.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    df = inp['data_processed'].copy()
    df_new = pandas.DataFrame()
    for batch in set(df['Batch']):
        df_batch = df[df['Batch'] == batch].copy()
        for group in set(df_batch['Group']):
            df_group = df_batch[df_batch['Group'] == group].copy()
            for comp in set(df_group['Component Name']):
                df_comp = df_group[df_group['Component Name'] == comp].copy()
                df_comp['zscore'] = scipy.stats.zscore(df_comp['Area'])
                df_new = df_new.append(df_comp)

    for i,row in df_new.iterrows():
        if abs(row['zscore'])>3:
            df_new.at[i,'Area'] = numpy.nan
            df_new.at[i,'Signal / Noise'] = numpy.nan
    df_new = df_new.reset_index(drop = True)
    inp['data_processed'] = df_new.copy()
    # Save dataframe
    save_df(df_new, inp['path_evaluation_identification'], f'03_outlier_SN{inp["pre_signal_noise"]}', index = True)
    return inp

def scan_preprocessing_imputation(inp):
    """
    Outlier processing.

    Preprocessing procedure for outlier handling.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Only for samples, not for QC
    # Allocation
    df = inp['data_processed_raw'].copy()
    # Missing at random
    df_mar = scan_preprocessing_imputation_mar(df)
    # Missing not at random
    df_mnar = scan_preprocessing_imputation_mnar(df_mar)
    # Allocate
    inp['data_processed'] = df_mnar.copy()
    return inp

def scan_preprocessing_imputation_mar(df):
    """
    Missing-At-Random.

    Outlier handling for Missing-At-Random values.

    Parameters
    ----------
    df : dataframe
        Dataframe with potentially MAR values.

    Returns
    -------
    df_mar : dataframe
        Dataframe with no potentially MAR values.
    """
    # Get samples, not QC
    groups_imputation = set([item for item in df['Group'] if item.upper() != 'QC'])
    # Mean imputation for Missing-At-Random (MAR)
    # Preallocation
    df_mar = df[df['Group'] == 'QC'].copy()
    # Cycle groups
    for group in groups_imputation:
        df_group = df[df['Group'] == group].copy()
        # Cycle compounds
        for compound in set(df_group['Component Name']):
            df_compound = df_group[df_group['Component Name'] == compound].copy()
            # Get series
            series_area = df_compound['Area'].copy()
            series_sn = df_compound['Signal / Noise'].copy()
            # Get missing data for MAR imputation
            series_area[pandas.isna(series_area)] = numpy.nanmean(series_area)
            series_sn[pandas.isna(series_area)] = numpy.nanmean(series_sn)
            df_compound['Area'] = series_area
            df_compound['Signal / Noise'] = series_sn
            # Append
            df_mar = df_mar.append(df_compound)
    df_mar = df_mar.sort_values('Order')
    return df_mar

def scan_preprocessing_imputation_mnar(df):
    """
    Missing-Not-At-Random.

    Outlier handling for Missing-Not-At-Random values.

    Parameters
    ----------
    df : dataframe
        Dataframe with potentially MNAR values.

    Returns
    -------
    df_mar : dataframe
        Dataframe with no potentially MNAR values.
    """
    # Get samples, not QC
    groups_imputation = set([item for item in df['Group'] if item.upper() != 'QC'])
    # Half minimum imputation for Missing-Not-At-Random
    # Preallocation
    df_mnar = df[df['Group'] == 'QC'].copy()
    # Cycle compounds
    df_groups = df[df['Group'].isin(groups_imputation)].copy()
    for compound in set(df_groups['Component Name']):
        df_compound = df_groups[df_groups['Component Name'] == compound].copy()
        # Get series
        series_area = df_compound['Area'].copy()
        series_sn = df_compound['Signal / Noise'].copy()
        # Get missing data for MNAR imputation
        series_area[pandas.isna(series_area)] = numpy.nanmin(series_area)/2
        series_sn[pandas.isna(series_area)] = numpy.nanmin(series_sn)/2
        df_compound['Area'] = series_area
        df_compound['Signal / Noise'] = series_sn
        # Append
        df_mnar = df_mnar.append(df_compound)
    df_mnar = df_mnar.sort_values('Order')
    return df_mnar

def scan_preprocessing_filter(inp):
    """
    Filter data.

    Label qualitative, quantitative and RSD information.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    df_filter : dataframe
        Labeled data.
    """
    # Get data
    df_filter = inp['data_processed'].copy()
    # Consider replicates
    df_filter = df_filter.groupby(['Group','Reactor','Sample Number','Dilution factor','Component Name','Batch','Class']).agg([numpy.nanmean, numpy.nanstd, 'count', 'size']).copy().reset_index()
    # Get qualitative
    df_filter[f'qualitative'] = df_filter['Signal / Noise','nanmean'] >= inp['pre_signal_noise']
    # Get abundance
    df_filter[f'missing'] = 1-(df_filter['Area','count']/df_filter['Area','size'])
    # Get quantitative
    df_filter['RSD'] = (df_filter['Area', 'nanstd']/df_filter['Area', 'nanmean'])*100
    df_filter[f'quantitative'] = df_filter['RSD'] <= inp['pre_rsd']
    df_filter['identification'] = numpy.where(df_filter['quantitative'], 'quantitative', 'qualitative')
    # Apply minimum S/N filter
    df_filter = df_filter[df_filter['qualitative'] == True].copy()
    return df_filter

def scan_preprocessing_create_mapper_report(inp):
    """
    Create report mapper.

    Create mapper for names, modes, isobarics and references.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Allocate
    dict_report = inp['information_report'].copy()
    # Create mapper for comparison
    inp['mapper_name'] = dict(zip(dict_report['single']['compound_id'], dict_report['single']['compound_name']))
    inp['mapper_mode'] = dict(zip(dict_report['convoluted']['compound_id'], dict_report['convoluted']['mode']))
    inp['mapper_isobarics'] = dict(zip(dict_report['convoluted']['compound_id'], dict_report['convoluted']['isobarics avoided']))
    inp['mapper_reference'] = dict(zip(dict_report['convoluted']['compound_id'], dict_report['convoluted']['reference']))
    return inp

def scan_preprocessing_create_mapper_group(inp):
    """
    Create group mapper.

    Create mapper for group, colors and marker.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    df = inp['data_processed_raw'].copy()
    groups_samples = list(pandas.Series([item for item in sorted(df['Group']) if 'QC ' not in item]).unique())
    groups_qc = list(pandas.Series([item for item in sorted(df['Group']) if 'QC ' in item]).unique())
    series_groups = groups_samples + groups_qc
    inp['mapper_nr_group'] = dict(enumerate(series_groups))
    inp['mapper_group_nr'] = {y:x for x,y in inp['mapper_nr_group'].items()}
    inp['palette_group'] = seaborn.color_palette('colorblind', n_colors = len(groups_samples))+len(groups_qc)*['k']
    inp['marker_group'] = markerselection(n_marker = len(groups_samples)+len(groups_qc))
    inp['mapper_group_color'] = {x:inp['palette_group'][y] for x,y in inp['mapper_group_nr'].items()}
    inp['mapper_group_marker'] = {x:inp['marker_group'][y] for x,y in inp['mapper_group_nr'].items()}
    return inp

def scan_preprocessing_reference(df, inp):
    """
    Add references.

    Label data with sources, mode and convolution.

    Parameters
    ----------
    df : dataframe
        Unlabeled data.
    inp : dict
        Method dictionary.

    Returns
    -------
    df_new : dataframe
        Labeled data.
    """
    df_new = df.copy()
    df_new['mode'] = df_new['Component Name'].map(inp['mapper_mode'])
    df_new['isobarics avoided'] = df_new['Component Name'].map(inp['mapper_isobarics'])
    df_new['reference'] = df_new['Component Name'].map(inp['mapper_reference'])
    # Replace values
    list1 = ['IBG1','inorganic']
    hold1 = df_new[df_new['reference'].isin(list1)]
    df_new.loc[hold1.index,'reference'] = 'inhouse'
    list2 = ['predicted','inhouse']
    hold2 = df_new[~df_new['reference'].isin(list2)]
    df_new.loc[hold2.index,'reference'] = 'literature'
    df_new['isobarics avoided'] = numpy.where(df_new['isobarics avoided'], 'unique', 'convolution')
    return df_new

def scan_preprocessing_cluster(df, inp):
    """
    Cluster data.

    Get group sets for data.

    Parameters
    ----------
    df : dataframe
        Unlabeled data.
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Preallocate
    df_cluster = df.copy()

    # All sets
    dict_sets = {}
    set_list = []

    df = df[df['Class']!='QC'].copy()
    df_cluster = df_cluster[df_cluster['Class']!='QC'].copy()
    for model in set(df['Group']):
        df_model = df[df['Group'] == model].copy()
        set_hold = set(df_model['Component Name'])
        dict_sets[model] = set_hold
        set_list.append(set_hold)
    # Intersect of all sets
    intersect = set.intersection(*set_list)
    index_intersect = df[df['Component Name'].isin(intersect)].index
    df_cluster.loc[index_intersect,'set'] = 'intersect'
    df.loc[index_intersect,'set'] = 'intersect'
    # Unique of specific set
    dict_unique = {}
    keys = set(dict_sets.keys())
    for model in keys:
        set_i = dict_sets[model]
        keys_rest = keys - {model}
        list_rest = []
        for key in keys_rest:
            list_rest.append(dict_sets[key])
        dict_unique[model] = set_i.difference(*list_rest)

    # Mark dataframe for model unique
    for key in dict_unique.keys():
        index_unique = df[df['Component Name'].isin(dict_unique[key])].index
        df_cluster.loc[index_unique,'set'] = 'unique'   
        df.loc[index_unique,'set'] = 'unique'
    # Mark dataframe for rest
    # Last NANs are for example in A and B but not C, therefore no Intersect but also not unique
    index_nunique = df[df['set'].isna()].index
    df_cluster.loc[index_nunique,'set'] = 'subset'
    df.loc[index_nunique,'set'] = 'subset'
    # Allocate dataframe
    inp['data_cluster'] = df_cluster.copy()
    return inp

def scan_preprocessing_plotting(inp, variables, filename):
    """
    Preprocessing plots.

    Create plots for preprocessed data.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    variables : list
        List of variables from clustering.
    filename : str
        Filename for plots.
    """
    # Create copy
    df_plot = inp['data_cluster'].copy()
    df_plot.columns = df_plot.columns.droplevel(1)
    # Drop nan
    #df_plot = df_plot.dropna(how = 'any')
    # Create palette
    palette = create_palette(8, reverse = False)
    # Set legend properties
    legendproperties = {'size': inp['pre_labelsize'],'weight': 'bold'}
    # Create plots
    if filename == 'cluster':
        figsize = inp['pre_figsize_identification']
    else:
        figsize = inp['pre_figsize_quantification']
    

    fig, axes = plt.subplots(figsize = figsize, ncols = len(variables) , sharex = False, sharey = True)
    # Create labels for subplots
    list_pic_all = [chr(x) for x in range(ord('a'), ord('z') + 1)]
    list_pic = list_pic_all[:len(axes.flatten())]
    
    leg = []
    # Cycle variables
    for variable, ax, marker in zip(variables, axes, list_pic):
        ax.text(0.05, 0.95, marker, 
            fontsize = inp['pre_labelsize']+2, fontweight = 'bold', color = 'black', bbox = dict(facecolor='white', edgecolor='black'),
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        # Preallocate bar lists
        bars_ref_1 = []
        bars_ref_2 = []
        bars_ref_3 = []
        # Preallocate name lists
        r = []
        names = []
        # Cycle models
        for i, model in enumerate(df_plot['Group'].unique()):
            df_model = df_plot[df_plot['Group'] == model].copy()
            df_model = df_model.drop_duplicates(subset=['Component Name'])
            # Values of each group
            if variable == 'reference':
                if 'inhouse' in df_model['reference'].unique():
                    bars_ref_1.append(df_model['reference'].value_counts()['inhouse'])
                else:
                    bars_ref_1.append(0)
                if 'literature' in df_model['reference'].unique():
                    bars_ref_2.append(df_model['reference'].value_counts()['literature'])
                else:
                    bars_ref_2.append(0)
                if 'predicted' in df_model['reference'].unique():
                    bars_ref_3.append(df_model['reference'].value_counts()['predicted'])
                else:
                    bars_ref_3.append(0)
            if variable == 'mode':
                if 'Pos' in df_model['mode'].unique():
                    bars_ref_1.append(df_model['mode'].value_counts()['Pos'])
                else:
                    bars_ref_1.append(0)
                if 'Neg' in df_model['mode'].unique():
                    bars_ref_2.append(df_model['mode'].value_counts()['Neg'])
                else:
                    bars_ref_2.append(0)
            if variable == 'isobarics':
                if 'unique' in df_model['isobarics avoided'].unique():
                    bars_ref_1.append(df_model['isobarics avoided'].value_counts()['unique'])
                else:
                    bars_ref_1.append(0)
                if 'convolution' in df_model['isobarics avoided'].unique():
                    bars_ref_2.append(df_model['isobarics avoided'].value_counts()['convolution'])
                else:
                    bars_ref_2.append(0)
            if variable == 'identification':
                if 'quantitative' in df_model['identification'].unique():
                    bars_ref_1.append(df_model['identification'].value_counts()['quantitative'])
                else:
                    bars_ref_1.append(0)
                if 'qualitative' in df_model['identification'].unique():
                    bars_ref_2.append(df_model['identification'].value_counts()['qualitative'])
                else:
                    bars_ref_2.append(0)
            if variable == 'sets':
                if 'intersect' in df_model['set'].unique():
                    bars_ref_1.append(df_model['set'].value_counts()['intersect'])
                else:
                    bars_ref_1.append(0)
                if 'subset' in df_model['set'].unique():
                    bars_ref_2.append(df_model['set'].value_counts()['subset'])
                else:
                    bars_ref_2.append(0)
                if 'unique' in df_model['set'].unique():
                    bars_ref_3.append(df_model['set'].value_counts()['unique'])
                else:
                    bars_ref_3.append(0)
            # The position of the bars on the x-axis
            r.append(i)
            names.append(model)
        # Heights of bar1 + bar2
        bars = numpy.add(bars_ref_1, bars_ref_2).tolist()
        # Names of group and bar width
        barWidth = 1
        # Values of each group
        if variable == 'reference':
            if 'inhouse' in df_plot['reference'].unique():
                # Create bottom bars
                ax.bar(r, bars_ref_1, color = palette[0], edgecolor='white', label = 'inhouse')
            if 'literature' in df_plot['reference'].unique():
                # Create middle bars, on top of the firs ones
                ax.bar(r, bars_ref_2, bottom = bars_ref_1, color=palette[1], edgecolor='white', label = 'literature')
            if 'predicted' in df_plot['reference'].unique():
                # Create top bars
                ax.bar(r, bars_ref_3, bottom = bars, color=palette[2], edgecolor='white', label = 'prediction')
        if variable == 'mode':
            if 'Pos' in df_plot['mode'].unique():
                # Create bottom bars
                ax.bar(r, bars_ref_1, color = palette[0], edgecolor='white', label = 'positive')
            if 'Neg' in df_plot['mode'].unique():
                # Create middle bars, on top of the firs ones
                ax.bar(r, bars_ref_2, bottom = bars_ref_1, color=palette[1], edgecolor='white', label = 'negative')
        if variable == 'isobarics':
            if 'unique' in df_plot['isobarics avoided'].unique():
                # Create bottom bars
                ax.bar(r, bars_ref_1, color = palette[0], edgecolor='white', label = 'unique')
            if 'convolution' in df_plot['isobarics avoided'].unique():
                # Create middle bars, on top of the firs ones
                ax.bar(r, bars_ref_2, bottom = bars_ref_1, color=palette[1], edgecolor='white', label = 'convolution')
        if variable == 'identification':
            if 'quantitative' in df_plot['identification'].unique():
                # Create bottom bars
                ax.bar(r, bars_ref_1, color = palette[0], edgecolor='white', label = 'quantification')
            if 'qualitative' in df_plot['identification'].unique():
                # Create middle bars, on top of the firs ones
                ax.bar(r, bars_ref_2, bottom = bars_ref_1, color=palette[1], edgecolor='white', label = 'qualification')
        if variable == 'sets':
            if 'intersect' in df_plot['set'].unique():
                # Create bottom bars
                ax.bar(r, bars_ref_1, color = palette[0], edgecolor='white', label = 'intersect')
            if 'subset' in df_plot['set'].unique():
                # Create middle bars, on top of the firs ones
                ax.bar(r, bars_ref_2, bottom = bars_ref_1,  color=palette[1], edgecolor='white', label = 'subset')
            if 'unique' in df_plot['set'].unique():
                # Create top bars
                ax.bar(r, bars_ref_3, bottom = bars, color=palette[2], edgecolor='white', label = 'unique')
        # Set x ticks
        ax.set_xticks(r)
        ax.set_xticklabels(labels = names, size = inp['pre_labelsize'], rotation = '35', ha = 'right')
        # Set labels
        ax.set_xlabel('Group', size = inp['pre_labelsize'], fontweight = 'bold')
        if ax.is_first_col():
            ax.set_ylabel('counts', size = inp['pre_labelsize'], fontweight = 'bold')
        else:
            ax.set_ylabel(None)
        # Resize ticks
        ax.tick_params(axis='both', labelsize = inp['pre_labelsize'])
        # Add legend
        leg.append(ax.legend(bbox_to_anchor=(1.0, 1.0),loc="lower right", frameon = False, prop = legendproperties))
        # Annotate bars
        for patch in ax.patches:
            # Get height of patch as label text
            value = patch.get_height()
            if value != 0:
                # Get x position of patch
                x_pos = patch.get_x() + patch.get_width() / 2.
                # Get y position of patch
                y_pos = patch.get_y() + patch.get_height() / 2.
                # Set label
                label = ax.annotate(value, (x_pos, y_pos), va = 'center', ha = 'center', fontsize = inp['pre_labelsize'], fontweight = 'bold') 
    fig.savefig(inp['path_evaluation_identification'].joinpath(f'{filename}_SN{inp["pre_signal_noise"]}.png'), bbox_extra_artists = (leg), bbox_inches = 'tight', dpi = 800)
    fig.savefig(inp['path_evaluation_identification'].joinpath(f'{filename}_SN{inp["pre_signal_noise"]}.svg'), bbox_extra_artists = (leg), bbox_inches = 'tight', format = 'svg')
    plt.close(fig)
    return 

def scan_preprocessing_identification(inp):
    """
    Identify substances.

    Identify substances of experiment. 

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Allocate
    df_identification = inp['data_cluster'].copy()
    # Drop multiindex
    df_identification.columns = df_identification.columns.droplevel(1)
    # Drop duplicate columns
    df_identification = df_identification.loc[:,~df_identification.columns.duplicated()]
    # Get single mass transitions
    df_identification = scan_expand_convolution(df_identification, inp)
    # Filter and sort important identification columns
    df_identification = df_identification.filter(
        ['Group', 'compound_id', 'compound_name', 'isobarics avoided', 'Reactor', 'Sample Number', 'Dilution factor',
        'Mode', 'Signal / Noise', 'RSD', 'qualitative', 'quantitative', 'reference','set']
        )
    # Save dataframe
    save_df(df_identification, inp['path_evaluation_identification'], f'identification_SN{inp["pre_signal_noise"]}', index = False)
    # Allocate dataframe
    inp['data_identification'] = df_identification.copy()
    return inp

def scan_preprocessing_filter_quantitative(inp):
    """
    Filter data.

    Filter data for univariate analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Preallocate data
    dict_data = {}
    # Allocate data
    df_cluster = inp['data_cluster'].copy()
    df_processed = inp['data_processed'].copy()
    # Get unique mass transitions
    df_cluster = scan_get_single_masstransitions(df_cluster)
    df_processed = scan_get_single_masstransitions(df_processed)
    # Allocate to dictionary
    dict_data['cluster'] = df_cluster.copy()
    dict_data['processed'] = df_processed.copy()
    # Apply RSD filter
    df_cluster = df_cluster[df_cluster['quantitative'] == True].copy()
    df_processed = df_processed[
        (df_processed['Component Name'].isin(set(df_cluster['Component Name'])))&
        (df_processed['Batch'].isin(set(df_cluster['Batch'])))
        ].copy()
    # Allocate dataframe
    inp['data_cluster_filter'] = df_cluster.copy()
    inp['data_processed_filter'] = df_processed.sort_values(['Group'], ascending = True).copy()
    # Allocate to dictionary
    dict_data['cluster_filter'] = df_cluster.copy()
    dict_data['processed_filter'] = df_processed.copy()
    # Save data
    save_df(df_processed, inp['path_evaluation_identification'], f'04_quantitative_SN{inp["pre_signal_noise"]}', index = False)
    save_dict(dict_data, inp['path_evaluation_identification'], f'05_cluster_SN{inp["pre_signal_noise"]}', single_files = False, index = True)
    return inp

def scan_uv(inp, uv_alpha_univariate = 0.05, uv_fold_change = 1, uv_decision_tree = True, uv_paired_samples = False, uv_correction = 'Holm-Bonferroni', uv_labelsize_vulcano = 12, uv_figsize_vulcano = (8,8), uv_label_full_vulcano = True):
    """
    Univariate analysis.

    Basis function for univariate statistics of scan data.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    uv_alpha_univariate : float
        Probability of error.
    uv_fold_change : float
        Fold change threshold.
    uv_decision_tree : bool
        Use univariate decision tree.
    uv_paired_samples : bool
        Dependent or independent samples.
    uv_correction : str
        Multi comparison correction.
    uv_labelsize_vulcano : int
        Labelsize for vulcano plot.
    uv_figsize_vulcano : tuple
        Figsize for vulcano plot.
    uv_label_full_vulcano : bool
        Use full labels in vulcano plot.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Multiple groups
    if len(set(inp['data_processed_filter']['Group'])) > 1:
        # Parse parameter
        inp = scan_uv_parsing(inp, uv_alpha_univariate, uv_fold_change, uv_decision_tree, uv_paired_samples, uv_correction, uv_labelsize_vulcano, uv_figsize_vulcano, uv_label_full_vulcano)
        # Create folder
        inp = scan_uv_folder(inp)
        # Quantification
        inp = scan_uv_main(inp)
    return inp

def scan_uv_parsing(inp, uv_alpha_univariate, uv_fold_change, uv_decision_tree, uv_paired_samples, uv_correction, uv_labelsize_vulcano, uv_figsize_vulcano, uv_label_full_vulcano):
    """
    Initialize univariate analysis.

    Initialization function for univariate analysis.

    inp : dict
        Method dictionary.
    uv_alpha_univariate : float
        Probability of error.
    uv_fold_change : float
        Fold change threshold.
    uv_decision_tree : bool
        Use univariate decision tree.
    uv_paired_samples : bool
        Dependent or independent samples.
    uv_correction : str
        Multi comparison correction.
    uv_labelsize_vulcano : int
        Labelsize for vulcano plot.
    uv_figsize_vulcano : tuple
        Figsize for vulcano plot.
    uv_label_full_vulcano : bool
        Use full labels in vulcano plot.

    Returns
    -------
    inp : dict
        Method dictionary.
    """

    # Parameter
    inp['uv_fold_change'] = uv_fold_change
    inp['uv_alpha'] = uv_alpha_univariate
    inp['uv_correction'] = uv_correction
    inp['uv_decision_tree'] = uv_decision_tree
    inp['uv_paired_samples'] = uv_paired_samples
    # Plots
    inp['uv_labelsize_vulcano'] = uv_labelsize_vulcano
    inp['uv_figsize_vulcano'] = uv_figsize_vulcano
    inp['uv_label_vulcano'] = uv_label_full_vulcano
    return inp

def scan_uv_folder(inp):
    """
    Create folder.

    Create folder for univariate analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    inp['path_evaluation_uv'] = create_folder(inp['path_evaluation'], '02_univariate')
    return inp

def scan_uv_main(inp):
    """
    Calculate univariate statistics.

    Univariate analysis and multi comparison correction.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Get classes
    inp = scan_uv_comparisons(inp)
    # Hypothesis tests
    inp = scan_uv_ttest(inp)
    # Vulcano plot
    if inp['pre_list_relevant']:
        scan_uv_plot(inp, bool_relevant = True)
    else:
        scan_uv_plot(inp, bool_relevant = False)
    return inp

def scan_uv_comparisons(inp):
    """
    Get comparisons.

    Get group comparison pairs for further analysis

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    df = inp['data_cluster_filter'].copy()
    df_samples = df[df['Class'] == 'Sample'].copy()
    # Get comparisons
    inp['groups'] = [item for item in df_samples['Group'].unique()]
    inp['comparisons'] = get_combinations([item for item in df_samples['Group'].unique()])
    return inp

def scan_uv_ttest(inp):
    """
    Univariate ttest.

    Calculate univariate center, variance and normality tests for metabolites of group in pairs.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Allocate data
    df_uv = inp['data_processed_filter'].copy()
    df_cluster_filter = inp['data_cluster_filter'].copy()

    # Preallocate data for statistics
    df_statistics_p = pandas.DataFrame()
    list_kruskal = []
    list_posthoc = []

    # Cycle analytes
    for analyte in df_cluster_filter['Component Name'].unique():
        df_cluster_analyte = df_cluster_filter[df_cluster_filter['Component Name'] == analyte].copy()
        
        # Kruskal-Wallis
        test_all = 'Kruskal-Wallis'
        df_processed_kruskal = df_uv[
            (df_uv['Component Name'] == analyte) &
            (df_uv['Class'] != 'QC')
            ].copy()
        df_processed_kruskal_list = df_processed_kruskal.groupby(['Group']).agg(lambda x: list(x))
        list_kruskal_raw = [numpy.array(item) for item in df_processed_kruskal_list['Area']]
        list_kruskal_nanpolicy = [item[~numpy.isnan(item)] for item in list_kruskal_raw]
        stat_all, p_value_all = scipy.stats.kruskal(*list_kruskal_nanpolicy)
        list_kruskal.append([analyte, test_all, stat_all, p_value_all])

        # Post-hoc tests
        if len(set(df_cluster_analyte['Group'])) > 1:

            # Get combinations
            pairs = inp['comparisons']

            # Post-hoc decision tree
            if inp['uv_decision_tree'] == True:
                # Cycle combinations
                for pair in pairs:
                    # Select data
                    df_data = df_uv[
                        (df_uv['Group'].isin(list(pair))) & 
                        (df_uv['Component Name'] == analyte)
                        ].copy().reset_index()
                    df1 = df_data[(df_data['Group'] == pair[0]) & (~pandas.isna(df_data['Area']))].copy()
                    df2 = df_data[(df_data['Group'] == pair[1]) & (~pandas.isna(df_data['Area']))].copy()

                    # Normality
                    test_normality = 'Shapiro-Wilk'
                    try:
                        stat_normality_1, p_value_normality_1 = scipy.stats.shapiro(df1['Area'])
                        stat_normality_2, p_value_normality_2 = scipy.stats.shapiro(df2['Area'])
                    except:
                        print(df1)
                        print(df2)
                    significant_normality_group1 = p_value_normality_1 < 0.05
                    significant_normality_group2 = p_value_normality_2 < 0.05
                    # Normal distribution of both sets
                    if (significant_normality_group1 == False) | (significant_normality_group2 == False):                        
                        # Bartlett's test
                        test_variance = 'Bartlett'
                        stat_variance, p_value_variance = scipy.stats.bartlett(df1['Area'], df2['Area'])

                        # Mean
                        if inp['uv_paired_samples'] == True:
                            # T-test dependent
                            test_center = 't-test dependent'
                            n_1 = len(df1['Area'].dropna())
                            n_2 = len(df2['Area'].dropna())
                            
                            if n_1 != n_2:
                                n_paired = numpy.nanmin([n_1, n_2])
                                array_1 = numpy.array(df1.loc[:n_paired-1,'Area'])
                                array_2 = numpy.array(df2.loc[:n_paired-1,'Area'])
                            else:
                                array_1 = numpy.array(df1['Area'])
                                array_2 = numpy.array(df2['Area'])
                            stat_center, p_value_center = scipy.stats.ttest_rel(array_1, array_2)
                        else:
                            if p_value_variance < 0.05:
                                # Welch's test
                                test_center = 'Welch'
                                stat_center, p_value_center = scipy.stats.ttest_ind(df1['Area'], df2['Area'], equal_var = False)
                            else:
                                # T-test independent
                                test_center = 't-test independent'
                                stat_center, p_value_center = scipy.stats.ttest_ind(df1['Area'], df2['Area'], equal_var = True)
                            
                    # Non normal distribution of at least one set
                    else:
                        # Levene's test
                        test_variance = 'Levene'
                        stat_variance, p_value_variance = scipy.stats.levene(df1['Area'], df2['Area'])
                        # Median
                        if inp['uv_paired_samples'] == True:
                            test_center = 'Wilcoxon signed-rank'
                            stat_center, p_value_center = scipy.stats.wilcoxon(df1['Area'], df2['Area'])
                        else:
                            test_center = 'Wilcoxon rank-sum'
                            stat_center, p_value_center = scipy.stats.ranksums(df1['Area'], df2['Area'])
                    
                    # Fold change
                    fc = numpy.nanmean(df1['Area'])/numpy.nanmean(df2['Area'])

                    list_posthoc.append(
                        [
                            analyte, pair[0], pair[1], pair[0]+'_over_'+pair[1], fc, 
                            test_center, stat_center, p_value_center,
                            test_variance, stat_variance, p_value_variance,
                            test_normality, stat_normality_1, p_value_normality_1, stat_normality_2, p_value_normality_2
                        ]
                        )
                    df_statistics_p = df_statistics_p.append(df1)
                    df_statistics_p = df_statistics_p.append(df2)
            else:
                # Cycle combinations
                for pair in pairs:
                    # Select data
                    df_data = df_uv[
                        (df_uv['Group'].isin(list(pair))) & 
                        (df_uv['Component Name'] == analyte)
                        ].copy()
                    df1 = df_data[(df_data['Group'] == pair[0]) & (~pandas.isna(df_data['Area']))].copy().reset_index()
                    df2 = df_data[(df_data['Group'] == pair[1]) & (~pandas.isna(df_data['Area']))].copy().reset_index()

                    # Normality
                    test_normality = 'Shapiro-Wilk'
                    try:
                        stat_normality_1, p_value_normality_1 = scipy.stats.shapiro(df1['Area'])
                        stat_normality_2, p_value_normality_2 = scipy.stats.shapiro(df2['Area'])
                    except:
                        print(df1)
                        print(df2)
                    significant_normality_group1 = p_value_normality_1 < 0.05
                    significant_normality_group2 = p_value_normality_2 < 0.05
                                          
                    # Bartlett's test
                    test_variance = 'Bartlett'
                    stat_variance, p_value_variance = scipy.stats.bartlett(df1['Area'], df2['Area'])

                    # Mean
                    if inp['uv_paired_samples'] == True:
                        # T-test dependent
                        test_center = 't-test dependent'
                        n_1 = len(df1['Area'].dropna())
                        n_2 = len(df2['Area'].dropna())
                        
                        if n_1 != n_2:
                            n_paired = numpy.nanmin([n_1, n_2])
                            array_1 = numpy.array(df1.loc[:n_paired-1,'Area'])
                            array_2 = numpy.array(df2.loc[:n_paired-1,'Area'])
                        else:
                            array_1 = numpy.array(df1['Area'])
                            array_2 = numpy.array(df2['Area'])
                        stat_center, p_value_center = scipy.stats.ttest_rel(array_1, array_2)
                    else:
                        # T-test independent
                        test_center = 't-test independent'
                        stat_center, p_value_center = scipy.stats.ttest_ind(df1['Area'], df2['Area'], equal_var = True)
                        
                    # Fold change
                    fc = numpy.nanmean(df1['Area'])/numpy.nanmean(df2['Area'])

                    list_posthoc.append(
                        [
                            analyte, pair[0], pair[1], pair[0]+'_over_'+pair[1], fc, 
                            test_center, stat_center, p_value_center,
                            test_variance, stat_variance, p_value_variance,
                            test_normality, stat_normality_1, p_value_normality_1, stat_normality_2, p_value_normality_2
                        ]
                        )
                    df_statistics_p = df_statistics_p.append(df1)
                    df_statistics_p = df_statistics_p.append(df2)
    # Create dataframes
    df_statistics_test_kruskal = pandas.DataFrame(
        list_kruskal, columns = ['Component Name', 'Test center', 'Statistic center', 'p center'])
    df_statistics_test_kruskal['significant center'] = df_statistics_test_kruskal['p center'] < inp['uv_alpha']

    df_statistics_test_posthoc = pandas.DataFrame(
        list_posthoc, columns = [
            'Component Name', 'group1', 'group2', 'Relation', 'FC', 
            'Test center','Statistic center', 'p center', 
            'Test variance','Statistic variance', 'p variance', 
            'Test normality','Statistic normality group1', 'p normality group1', 'Statistic normality group2', 'p normality group2'
            ]
            )

    # Multiple testing correction
    df_statistics_test_posthoc_corrected = pandas.DataFrame()
    # Cycle analytes
    for analyte in df_statistics_test_posthoc['Component Name'].unique():
        df_analyte = df_statistics_test_posthoc[df_statistics_test_posthoc['Component Name'] == analyte].copy()
        # Multiple testing correction
        df_analyte['p center adjusted'] = multiple_testing_correction(df_analyte['p center'], correction_type = inp['uv_correction'])
        df_analyte['p variance adjusted'] = multiple_testing_correction(df_analyte['p variance'], correction_type = inp['uv_correction'])
        # Check for significance
        df_analyte['significant center'] = df_analyte['p center adjusted'] < inp['uv_alpha']
        df_analyte['significant variance'] = df_analyte['p variance adjusted'] < inp['uv_alpha']
        # Calculate plotting parameter
        df_analyte['log2(FC)'] = numpy.log2(df_analyte['FC'])
        df_analyte['-log10(p)'] = -numpy.log10(df_analyte['p center adjusted'])
        # Append dataframe
        df_statistics_test_posthoc_corrected = df_statistics_test_posthoc_corrected.append(df_analyte)

    if inp['uv_paired_samples'] == True: 
        append_paired = '_paired'
    else:
        append_paired = '_non-paired'

    if inp['uv_decision_tree'] == True:
        append_tree = '_tree'
    else:
        append_tree = '_no-tree'

    save_df(df_statistics_test_kruskal, inp['path_evaluation_uv'], f'data_kruskal_SN{inp["pre_signal_noise"]}{append_paired}{append_tree}', index = False)
    save_df(df_statistics_test_posthoc_corrected, inp['path_evaluation_uv'], f'data_posthoc_SN{inp["pre_signal_noise"]}{append_paired}{append_tree}', index = False)
    save_df(df_statistics_p, inp['path_evaluation_uv'], f'data_p_SN{inp["pre_signal_noise"]}{append_paired}{append_tree}', index = False)
    
    inp['uv_statistic_kruskal'] = df_statistics_test_kruskal.copy()
    inp['uv_statistic_posthoc'] = df_statistics_test_posthoc_corrected.copy()
    inp['data_statistic'] = df_statistics_p.copy()

    return inp

def scan_uv_plot(inp, bool_relevant):
    """
    Vulcano plot.

    Create vulcano plots with probability and fold change.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    bool_relevant : bool
        Only label list of relevant metabolites provided by user.
    """
    df_statistics_test = inp['uv_statistic_posthoc'].copy()

    # Cycle relations
    for relation in set(df_statistics_test['Relation']):
        df_relation = df_statistics_test[df_statistics_test['Relation'] == relation]
        cutoff_fc = inp['uv_fold_change']
        cutoff_alpha = inp['uv_alpha']
        sets = ['fcvalid','pvalid','invalid']
        palette = ['green','blue','grey']

        # Create plot
        fig = plt.figure(figsize = inp['uv_figsize_vulcano'])
        # Add plots for multiple axes
        ax1=fig.add_subplot(111, label="1", frameon = True)
        ax2=fig.add_subplot(111, label="2", frameon = False)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

        df_relation = splitDataFrameList(df_relation, 'Component Name', '_').copy()
        for comp in df_relation['Component Name']:
            df_temp = df_relation[df_relation['Component Name'] == comp].copy()
            df_relation.loc[df_temp.index, '-log10(p)'] = df_temp.loc[df_temp.index, '-log10(p)'] + [random.uniform(a = 0.001, b = 0.002) for item in range(len(df_temp['-log10(p)']))]
            df_relation.loc[df_temp.index, 'log2(FC)'] = df_temp.loc[df_temp.index, 'log2(FC)'] + [random.uniform(a = 0.001, b = 0.002) for item in range(len(df_temp['log2(FC)']))]

        texts = []
        for prop, col in zip(sets, palette):
            if prop == 'fcvalid':
                df_select = df_relation[
                    ((df_relation['log2(FC)'] < -numpy.log2(cutoff_fc)) | 
                    (df_relation['log2(FC)'] > numpy.log2(cutoff_fc))) &
                    (df_relation['-log10(p)'] > -numpy.log10(cutoff_alpha))
                ].copy().reset_index()
                alpha = 0.8
                if bool_relevant == False:
                    if inp['uv_label_vulcano'] == True:
                        texts += [ax1.text(df_select.at[i,'log2(FC)'], df_select.at[i,'-log10(p)'], inp['mapper_name'][df_select.at[i,"Component Name"]], color = 'black', fontsize =  inp['uv_labelsize_vulcano']-4) for i in range(len(df_select))]
                    else:
                        texts +=  [ax1.text(df_select.at[i,'log2(FC)'], df_select.at[i,'-log10(p)'], df_select.at[i,"Component Name"], color = 'black', fontsize =  inp['uv_labelsize_vulcano']-4) for i in range(len(df_select))]
                else:
                    df_relevant = df_select[df_select['Component Name'].isin(set(inp['pre_list_relevant']))].copy().reset_index()
                    if inp['uv_label_vulcano'] == True:
                        texts += [ax1.text(df_relevant.at[i,'log2(FC)'], df_relevant.at[i,'-log10(p)'], inp['mapper_name'][df_relevant.at[i,"Component Name"]], color = 'black', fontsize =  inp['uv_labelsize_vulcano']-2) for i in range(len(df_relevant))]
                    else:
                        texts += [ax1.text(df_relevant.at[i,'log2(FC)'], df_relevant.at[i,'-log10(p)'], df_relevant.at[i,"Component Name"], color = 'black', fontsize =  inp['uv_labelsize_vulcano']-2) for i in range(len(df_relevant))]

            elif prop == 'pvalid':
                df_select = df_relation[
                    ((df_relation['log2(FC)'] > -numpy.log2(cutoff_fc)) & 
                    (df_relation['log2(FC)'] < numpy.log2(cutoff_fc))) &
                    (df_relation['-log10(p)'] > -numpy.log10(cutoff_alpha))
                ].copy().reset_index()
                alpha = 0.3
                if bool_relevant == True:
                    df_relevant = df_select[df_select['Component Name'].isin(set(inp['pre_list_relevant']))].copy().reset_index()
                    if inp['uv_label_vulcano'] == True:
                        texts += [ax1.text(df_relevant.at[i,'log2(FC)'], df_relevant.at[i,'-log10(p)'], inp['mapper_name'][df_relevant.at[i,"Component Name"]], color = 'black', fontsize =  inp['uv_labelsize_vulcano']-2) for i in range(len(df_relevant))]
                    else:
                        texts += [ax1.text(df_relevant.at[i,'log2(FC)'], df_relevant.at[i,'-log10(p)'], df_relevant.at[i,"Component Name"], color = 'black', fontsize =  inp['uv_labelsize_vulcano']-2) for i in range(len(df_relevant))]
                else:
                    None

            else:
                df_select = df_relation[
                    ((df_relation['log2(FC)'] > -numpy.log2(cutoff_fc)) | 
                    (df_relation['log2(FC)'] < numpy.log2(cutoff_fc))) &
                    (df_relation['-log10(p)'] < -numpy.log10(cutoff_alpha))
                ].copy().reset_index()
                alpha = 0.4
                if bool_relevant == True:
                    df_relevant = df_select[df_select['Component Name'].isin(set(inp['pre_list_relevant']))].copy().reset_index()
                    if inp['uv_label_vulcano'] == True:
                        texts += [ax1.text(df_relevant.at[i,'log2(FC)'], df_relevant.at[i,'-log10(p)'], inp['mapper_name'][df_relevant.at[i,"Component Name"]], color = 'black', fontsize =  inp['uv_labelsize_vulcano']-2) for i in range(len(df_relevant))]
                    else:
                        texts += [ax1.text(df_relevant.at[i,'log2(FC)'], df_relevant.at[i,'-log10(p)'], df_relevant.at[i,"Component Name"], color = 'black', fontsize =  inp['uv_labelsize_vulcano']-2) for i in range(len(df_relevant))]
                else:
                    None
            seaborn.scatterplot(data = df_select, x = 'log2(FC)', y = '-log10(p)', s = inp['uv_labelsize_vulcano'], color = col, alpha = alpha, ax = ax1)
        
        limscaler_x = numpy.nanmax(abs(df_relation['log2(FC)']))
        ax1.set_xlim(-limscaler_x-limscaler_x*0.2, limscaler_x+limscaler_x*0.2)
        ax1.autoscale_view()
        plt.draw()

        adjustText.adjust_text(
            texts, ax = ax1, arrowprops=dict(arrowstyle='-', color='red', alpha = 0.5),
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05))                                       
        ax1.axhline(y = -numpy.log10(cutoff_alpha), linestyle = '--', linewidth = 0.8, color = 'black')
        if cutoff_fc > 1:
            ax1.axvline(x = -numpy.log2(cutoff_fc), linestyle = '--', linewidth = 0.8, color = 'black')
            ax1.axvline(x = numpy.log2(cutoff_fc), linestyle = '--', linewidth = 0.8, color = 'black')
        ax1.set_xlabel('log$_2$(FC)', fontsize = inp['uv_labelsize_vulcano'], fontweight = 'bold')
        ax1.set_ylabel('-log$_{10}$(p$_{adjusted}$)', fontsize = inp['uv_labelsize_vulcano'], fontweight = 'bold')
        ax1.tick_params(axis = 'both', labelsize = inp['uv_labelsize_vulcano'])
        
        # Create legend  
        list_parameter = [mlines.Line2D([],[], marker=None, linewidth = 0, label = f'alpha = {inp["uv_alpha"]}')]            
        if len(inp['comparisons'])>1:
            list_parameter.append(mlines.Line2D([],[], marker=None, linewidth = 0, label = f'correction = {inp["uv_correction"]}'))
            
        if cutoff_fc > 1:
            list_parameter.append(mlines.Line2D([],[], marker=None, linewidth = 0, label = f'fold change = {inp["uv_fold_change"]}'))

        list_combo = [
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f"{relation.replace('_',' ')}")
        ]
        legendproperties = {'size': inp['uv_labelsize_vulcano']-2, 'weight': 'bold'}
        leg1 = ax1.legend(handles = list_parameter, bbox_to_anchor=(0.0, 1.0), handlelength = 0,  loc="lower left", frameon = False, prop = legendproperties)
        leg2 = ax2.legend(handles = list_combo, bbox_to_anchor=(1.0, 1.0), handlelength = 0,  loc="lower right", frameon = False, prop = legendproperties)
        

        if bool_relevant == True:
            string_append = '_relevant' 
        else:
            string_append = ''

        if inp['uv_paired_samples'] == True: 
            append_paired = '_paired'
        else:
            append_paired = '_non-paired'

        if inp['uv_decision_tree'] == True:
            append_tree = '_tree'
        else:
            append_tree = '_no-tree'

        fig.savefig(inp['path_evaluation_uv'].joinpath(f'{relation}_SN{inp["pre_signal_noise"]}_a{inp["uv_alpha"]}_FC{inp["uv_fold_change"]}{string_append}{append_paired}{append_tree}.png'), bbox_extra_artists = [leg1, leg2], bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_uv'].joinpath(f'{relation}_SN{inp["pre_signal_noise"]}_a{inp["uv_alpha"]}_FC{inp["uv_fold_change"]}{string_append}{append_paired}{append_tree}.svg'), bbox_extra_artists = [leg1, leg2], bbox_inches = 'tight', format = 'svg')
        plt.close(fig)
    return

def scan_mv(inp, mv_scaling = True, mv_scaling_method = 'auto', mv_cv_iterator = 'kfold', mv_cv_stratified = True, mv_cv_repeated = True, mv_cv_kfold = 5, mv_cv_repetition = 2, mv_labelsize_mv = 12, mv_figsize_score = (8,8), mv_figsize_scree = (6,4), mv_figsize_vip = (5,10), mv_label_full_vip = True, mv_vip_number = 50):
    """
    Multivariate analysis.

    Basis function for multivariate analysis of scan data.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    mv_scaling : bool
        Apply scaling.
    mv_scaling_method : str
        If scaling is applied, select method.
    mv_cv_iterator : str
        Cross-validation method.
    mv_cv_stratified : bool
        Use stratification in CV.
    mv_cv_repeated : bool
        Use repetitions in CV.
    mv_cv_kfold : int
        Number of folds in k-Fold CV.
    mv_cv_repetition : int
        Number of repetitions of k-Fold CV.
    mv_labelsize_mv : int
        Labelsize for all plots.
    mv_figsize_score : tuple
        Figsize for scores plot.
    mv_figsize_scree : tuple
        Figsize for scree plot.
    mv_figsize_vip : tuple
        Figsize for VIP plot.
    mv_label_full_vip : bool
        Use full analyte name.
    mv_vip_number : int
        Number of VIP and Beta coefficients displayed.

    Returns
    -------
    inp : dict
        Method dictionary.
    """

    if len(inp['data_processed_filter']['Group'].unique()) > 1:
        # Parse parameter
        inp = scan_mv_parsing(inp, mv_scaling, mv_scaling_method, mv_cv_iterator, mv_cv_stratified, mv_cv_repeated, mv_cv_kfold, mv_cv_repetition, mv_labelsize_mv, mv_figsize_score, mv_figsize_scree, mv_figsize_vip, mv_label_full_vip, mv_vip_number)
        # Create folder
        inp = scan_mv_folder(inp)
        # Multivariate analysis
        inp = scan_mv_main(inp)
    return inp

def scan_mv_parsing(inp, mv_scaling, mv_scaling_method, mv_cv_iterator, mv_cv_stratified, mv_cv_repeated, mv_cv_kfold, mv_cv_repetition, mv_labelsize_mv, mv_figsize_score, mv_figsize_scree, mv_figsize_vip, mv_label_full_vip, mv_vip_number):
    """
    Initialize multivariate analysis.

    Initialization function for multivariate analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    mv_scaling : bool
        Apply scaling.
    mv_scaling_method : str
        If scaling is applied, select method.
    mv_cv_iterator : str
        Cross-validation method.
    mv_cv_stratified : bool
        Use stratification in CV.
    mv_cv_repeated : bool
        Use repetitions in CV.
    mv_cv_kfold : int
        Number of folds in k-Fold CV.
    mv_cv_repetition : int
        Number of repetitions of k-Fold CV.
    mv_labelsize_mv : int
        Labelsize for all plots.
    mv_figsize_score : tuple
        Figsize for scores plot.
    mv_figsize_scree : tuple
        Figsize for scree plot.
    mv_figsize_vip : tuple
        Figsize for VIP plot.
    mv_label_full_vip : bool
        Use full analyte name.
    mv_vip_number : int
        Number of VIP and Beta coefficients displayed.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Parameter
    inp['mv_scaling'] = mv_scaling
    inp['mv_scaling_method'] = mv_scaling_method
    inp['mv_cv_iterator'] = mv_cv_iterator
    inp['mv_cv_stratified'] = mv_cv_stratified
    inp['mv_cv_repeated'] = mv_cv_repeated
    inp['mv_cv_kfold'] = mv_cv_kfold
    inp['mv_cv_repetitions'] = mv_cv_repetition
    
    # Plots
    inp['mv_labelsize'] = mv_labelsize_mv
    inp['mv_figsize_score'] = mv_figsize_score
    inp['mv_figsize_scree'] = mv_figsize_scree
    inp['mv_figsize_vip'] = mv_figsize_vip
    inp['mv_label_vip'] = mv_label_full_vip
    inp['mv_vip_number'] = mv_vip_number
    return inp

def scan_mv_folder(inp):
    """
    Create folder.

    Create folder for multivariate analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    inp['path_evaluation_mv'] = create_folder(inp['path_evaluation'], '03_multivariate')
    return inp

def scan_mv_main(inp):
    """
    Calculate multivariate statistics.

    Multivariate modelling and analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    
    # Allocate data
    df_mv = inp['data_processed_filter'].copy()
    inp = scan_mv_modelling(df_mv, inp)
    return inp

def scan_mv_modelling(df, inp):
    """
    Multivariate modelling.

    Create multivariate models (PCA, PLS-DA) for scan data.

    Parameters
    ----------
    df : dataframe
        Scan data.
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """

    # Allocate data
    df_mv = df.copy()
    df_preprocessed = scan_mv_preprocessing(df_mv)

    # Sorting necessary to get y dummy type below, other arrangements will be automatically sorted (labelproblems) by crossdecomposition.plsregression
    # [
    #   [1,1,1,1,0,0,0,0,0,0,0,0],
    #   [0,0,0,0,1,1,1,1,0,0,0,0],
    #   [0,0,0,0,0,0,0,0,1,1,1,1],
    # ]

    df_preprocessed = df_preprocessed.sort_values(['Group']).reset_index(drop = True)
    inp['mapper_sample'] = dict(zip(df_preprocessed.index, df_preprocessed['Group']))
    # PCA
    scan_mv_pca(df_preprocessed, inp)
    # PLS-DA
    scan_mv_plsda(df_preprocessed, inp)
    return inp

def scan_mv_pca(df_preprocessed, inp):
    """
    PCA wrapper.

    Create PCA model and run evaluation.

    Parameters
    ----------
    df_preprocessed : dataframe
        Preprocessed data.
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Set variable type
    var_type = 'PC'
    # Select matrix X and vector y
    X = df_preprocessed[df_preprocessed.columns[~df_preprocessed.columns.isin(['Group'])]].values
    y = df_preprocessed['Group'].values
    # Get label
    labels_variable = df_preprocessed[df_preprocessed.columns[~df_preprocessed.columns.isin(['Group'])]].columns
    map_var = dict([(item, i) for (i, item) in enumerate(pandas.Series(labels_variable))])
    # Classify groups
    y_group = numpy.array([inp['mapper_group_nr'][item] for item in y])
    # Get label
    labels_group = pandas.Series(y).unique()
    # Get covariance matrix
    cov_x, u, v = scan_mv_covariance_matrix(X)
    # Find optimal number of PC
    df_pca_scree = scan_mv_optimization('pca', X, y_group, inp = inp)
    df_pca_scree, n_components = scan_mv_optimization_min(df_pca_scree, model_type = 'pca', inp = inp)
    scan_mv_optimization_scree_plot(df_pca_scree, 'pca', inp = inp)
    # Calculate pca with optimal parameter
    pca_fitted = scan_mv_pca_calculation_cv(X, y_group, n_components, inp)
    model_params = scan_mv_pca_model_params(pca_fitted, df_pca_scree)

    # Get data
    T = pca_fitted.scores
    P = numpy.transpose(pca_fitted.loadings)
    ex = numpy.sqrt(numpy.var(T, axis = 0))
    Pcorr = P*ex
    
    # Cycle PC pairs for plotting
    labels_pc = [var_type+str(item) for item in range(1,n_components+1)]
    combinations = get_combinations(labels_pc)

    # Create dataframes
    df_scores_train = scan_mv_scores_pca(pca_fitted, 'train', inp)
    df_scores_test = scan_mv_scores_pca(pca_fitted, 'test', inp)
    
    # Save pca data
    dict_pca = {
        'scree': df_pca_scree, 'model_params': pandas.DataFrame.from_dict(model_params, orient = 'index').reset_index(),
        'x_scores_train': df_scores_train, 'x_scores_test': df_scores_test,
        }

    # Name append    
    if inp['mv_scaling'] == True:
        app_scale = f'_{inp["mv_scaling_method"]}'
    else:
        app_scale = ''
    save_dict(dict_pca, inp['path_evaluation_mv'], f'data_pca_SN{inp["pre_signal_noise"]}{app_scale}', single_files = False, index = False)

    # Plotting
    variables_to_plot = ['PC1','PC2']
    for variables in combinations:
        if ((variables[0] in variables_to_plot)&(variables[1] in variables_to_plot)):
            scan_mv_scores_plot(df_scores_train, df_pca_scree, inp, model_params, variables, var_type, 'train')
            scan_mv_scores_plot(df_scores_test, df_pca_scree, inp, model_params, variables, var_type, 'test')

    return inp

def scan_mv_plsda(df_preprocessed, inp):
    """
    PLS-DA wrapper.

    Create PLS-DA model and run evaluation.

    Parameters
    ----------
    df_preprocessed : dataframe
        Preprocessed data.
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Set variable type
    var_type = 'LV'
    df_preprocessed = df_preprocessed[~df_preprocessed['Group'].str.contains('QC ')].copy()
    df_preprocessed = df_preprocessed.sort_values(['Group']).reset_index(drop = True)
    inp['mapper_sample'] = dict(zip(df_preprocessed.index, df_preprocessed['Group']))
    # Select matrix X and vector y
    X = df_preprocessed[df_preprocessed.columns[~df_preprocessed.columns.isin(['Group'])]].values
    y = df_preprocessed['Group'].values

    # Get label
    labels_variable = df_preprocessed[df_preprocessed.columns[~df_preprocessed.columns.isin(['Group'])]].columns
    map_var = dict([(item, i) for (i, item) in enumerate(pandas.Series(labels_variable))])

    # Classify groups
    y_group = numpy.array([inp['mapper_group_nr'][item] for item in y])
    
    # Get label
    labels_group = pandas.Series(y).unique()
    
    # Find optimal number of LV
    df_pls_scree = scan_mv_optimization('pls', X, y_group, inp = inp)
    df_pls_scree, n_components_x = scan_mv_optimization_min(df_pls_scree, model_type = 'pls', inp = inp)
    scan_mv_optimization_scree_plot(df_pls_scree, 'pls', inp = inp)
    
    # Calculate pls with optimal parameter
    pls_fitted = scan_mv_pls_calculation_cv(X, y_group, n_components_x, inp)
    model_params = scan_mv_pls_model_params(pls_fitted, df_pls_scree)

    # Get data
    T = pls_fitted.scores_t
    U = pls_fitted.scores_u
    P = pls_fitted.loadings_p
    Q = pls_fitted.loadings_q
    W = pls_fitted.weights_w
    C = pls_fitted.weights_c
    Ws = pls_fitted.rotations_ws
    Cs = pls_fitted.rotations_cs
    ex = numpy.sqrt(numpy.var(T, axis = 0))
    ey = numpy.sqrt(numpy.var(U, axis = 0))
    Pcorr = Ws * ex
    Qcorr = Q * ey

    # Cycle LV pairs for plotting
    lv_labels = [var_type+str(item) for item in range(1,n_components_x+1)]
    combinations = get_combinations(lv_labels)
    # Create dataframes
    df_scores_train_t = scan_mv_scores_pls(pls_fitted, 't','train', inp)
    df_scores_test_t = scan_mv_scores_pls(pls_fitted, 't','test', inp)
    df_scores_train_u = scan_mv_scores_pls(pls_fitted, 'u','train', inp)
    df_scores_test_u = scan_mv_scores_pls(pls_fitted, 'u','test', inp)

    # PLS plots
    df_beta = scan_mv_pls_beta(pls_fitted, labels_group, labels_variable)
    df_vip = scan_mv_pls_vip(pls_fitted, labels_group, labels_variable)
    dict_rocauc = scan_mv_pls_rocauc(pls_fitted, labels_group)

    inp = scan_mv_relevant_predictors(df_beta, df_vip, inp)
    scan_mv_pls_vip_plot_full(df_vip, labels_group, inp)
    scan_mv_pls_beta_plot_full(df_beta, labels_group, inp)

    # Diskriminant analysis plots
    scan_mv_permutation_plot(pls_fitted, labels_group, inp)
    scan_mv_pls_roc_plot_full(dict_rocauc, labels_group, inp)

    # Variable selection
    if inp['pre_list_relevant']:
        scan_mv_pls_vip_plot_relevant(df_vip, labels_group, inp)
        scan_mv_pls_beta_plot_relevant(df_beta, labels_group, inp)

    # Save pls data
    dict_pls = {
        'scree': df_pls_scree, 'model_params': pandas.DataFrame.from_dict(model_params, orient = 'index').reset_index(),
        'x_scores_train': df_scores_train_t, 'x_scores_test': df_scores_test_t, 
        'y_scores_train': df_scores_train_u, 'y_scores_test': df_scores_test_u,
        'VIP': df_vip, 'beta': df_beta,
        }

    # Name append    
    if inp['mv_scaling'] == True:
        app_scale = f'_{inp["mv_scaling_method"]}'
    else:
        app_scale = ''
    save_dict(dict_pls, inp['path_evaluation_mv'], f'data_pls_SN{inp["pre_signal_noise"]}{app_scale}', single_files = False, index = False)
    
    variables_to_plot = ['LV1','LV2']
    for variables in combinations:
        if ((variables[0] in variables_to_plot)&(variables[1] in variables_to_plot)):
            scan_mv_scores_plot(df_scores_train_t, df_pls_scree, inp, model_params, variables, var_type, 'train')
            scan_mv_scores_plot(df_scores_test_t, df_pls_scree, inp, model_params, variables, var_type, 'test')
    return

def scan_mv_permutation_plot(model, labels_group, inp):
    """
    Plot permutation test.

    Plotting function for permutation test with Q2Y, Missclassification and AUC.

    Parameters
    ----------
    model : object
        PLS-DA object.
    labels_group : list
        Group labels.
    inp : dict
        Method dictionary.
    """
    # Q2Y distribution
    fig, ax = plt.subplots(figsize = (8,5))
    x_q2y = model.permutationParameters['PLS']['Q2Y']
    ax.hist(x_q2y, histtype = 'stepfilled', edgecolor='k', alpha = 0.8)
    ax.axvline(numpy.nanmean(model.cvParameters['PLS']['Q2Y'], axis = 0))
    ax.set_xlabel('Q$^{2}$Y', fontsize = inp['mv_labelsize'], fontweight = 'bold')
    ax.set_ylabel('Count', fontsize = inp['mv_labelsize'], fontweight = 'bold')
    ax.tick_params(labelsize = inp['mv_labelsize'])
    # Name append    
    if inp['mv_scaling'] == True:
        app_scale = f'_{inp["mv_scaling_method"]}'
    else:
        app_scale = ''
    fig.savefig(inp['path_evaluation_mv'].joinpath(f'permutation_Q2Y_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_inches = 'tight', dpi = 800)
    fig.savefig(inp['path_evaluation_mv'].joinpath(f'permutation_Q2Y_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_inches = 'tight', format = 'svg')
    plt.close(fig)
    
    # Missclassification distribution
    fig, ax = plt.subplots(figsize = (8,5))
    x_miss = [len(item) for item in model.permutationParameters['DA']['Perm_TestMisclassifiedsamples']]
    ax.hist(x_miss, histtype = 'stepfilled', edgecolor='k', alpha = 0.8)
    ax.axvline(numpy.nanmean([len(item) for item in model.cvParameters['DA']['CV_TestMisclassifiedsamples']]))
    ax.set_xlabel('Missclassification', fontsize = inp['mv_labelsize'], fontweight = 'bold')
    ax.set_ylabel('Count', fontsize = inp['mv_labelsize'], fontweight = 'bold')
    ax.tick_params(labelsize = inp['mv_labelsize'])
    # Name append    
    if inp['mv_scaling'] == True:
        app_scale = f'_{inp["mv_scaling_method"]}'
    else:
        app_scale = ''
    fig.savefig(inp['path_evaluation_mv'].joinpath(f'permutation_MissClass_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_inches = 'tight', dpi = 800)
    fig.savefig(inp['path_evaluation_mv'].joinpath(f'permutation_MissClass_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_inches = 'tight', format = 'svg')
    plt.close(fig)

    # AUROC distribution
    if len(labels_group) > 2:
        for i, group in enumerate(labels_group):
            fig, ax = plt.subplots(figsize = (8,5))
            x_group = model.permutationParameters['DA']['Perm_TestAUC'][:,i]
            ax.hist(x_group, histtype = 'stepfilled', bins = numpy.linspace(0,1,10), edgecolor='k', alpha = 0.8, label = r'AUROC(CV,permuted)')
            ax.plot([],[], color='none', label = fr'p = {numpy.round(model.permutationParameters["p-values"]["TestAUC"],4)}')
            ax.axvline(numpy.nanmean(model.cvParameters['DA']['CV_TestAUC'], axis = 0)[i], color = 'k', label = r'AUROC(CV,original)')
            ax.set_xlim(-0.05,1.05)
            ax.set_xlabel('AUROC', fontsize = inp['mv_labelsize'], fontweight = 'bold')
            ax.set_ylabel('Count', fontsize = inp['mv_labelsize'], fontweight = 'bold')
            ax.tick_params(labelsize = inp['mv_labelsize'])
            # First Legend
            legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
            leg1 = ax.legend(loc="upper left", bbox_to_anchor = (1,1), frameon = False, prop = legendproperties)
            # Name append    
            if inp['mv_scaling'] == True:
                app_scale = f'_{inp["mv_scaling_method"]}'
            else:
                app_scale = ''
            list_bbox = [leg1]
            fig.savefig(inp['path_evaluation_mv'].joinpath(f'permutation_AUROC_{group}_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
            fig.savefig(inp['path_evaluation_mv'].joinpath(f'permutation_AUROC_{group}_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
            plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize = (8,5))
        x_group = model.permutationParameters['DA']['Perm_TestAUC'][:]
        ax.hist(x_group, histtype = 'stepfilled', bins = numpy.linspace(0,1,10), edgecolor='k', alpha = 0.5, label = r'AUROC(CV,permuted)')
        ax.plot([],[], color='none', label = fr'p = {numpy.round(model.permutationParameters["p-values"]["TestAUC"],4)}')
        ax.axvline(numpy.nanmean(model.cvParameters['DA']['CV_TestAUC'], axis = 0), color = 'k', label = r'AUROC(CV,original)')
        ax.set_xlim(-0.05,1.05)
        ax.set_xlabel('AUROC', fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax.set_ylabel('Count', fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax.tick_params(labelsize = inp['mv_labelsize'])

        # First Legend
        legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
        leg1 = ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon = False, prop = legendproperties)
        # Name append    
        if inp['mv_scaling'] == True:
            app_scale = f'_{inp["mv_scaling_method"]}'
        else:
            app_scale = ''
        list_bbox = [leg1]
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'permutation_AUROC_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'permutation_AUROC_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
        plt.close(fig)
    return

def scan_mv_scores_pca(model, cv_set, inp):
    """
    Extract PCA scores.

    Extract PCA scores and provide dataframe.

    Parameters
    ----------
    model : object
        PCA object.
    cv_set : str
        Train or test scores.
    inp : dict
        Method dictionary.

    Returns
    -------
    df_scores : dataframe
        Dataframe with scores.    
    """
    if cv_set.lower() == 'train':
        key = 'CV_TrainScores'
    else:
        key = 'CV_TestScores'
    list_scores_train = [(item[0],*item[1]) for item in itertools.chain.from_iterable(model.cvParameters[key])]
    df_scores = pandas.DataFrame(list_scores_train)
    df_scores['Group'] = df_scores[0].map(inp['mapper_sample'])
    components = [item for item in df_scores.columns if item not in [0, 'Group']]
    mapper_component = dict(zip([item for item in components], ['PC'+str(item) for item in components]))
    mapper_component[0] = 'Sample'
    df_scores.rename(columns=mapper_component, inplace = True)
    return df_scores

def scan_mv_scores_pls(model, matrix, cv_set, inp):
    """
    Extract PLS-DA scores.

    Extract PLS-DA scores and provide dataframe.

    Parameters
    ----------
    model : object
        PLS-DA object.
    matrix : str
        Predictor or response scores.
    cv_set : str
        Train or test scores.
    inp : dict
        Method dictionary.

    Returns
    -------
    df_scores : dataframe
        Dataframe with scores.    
    """
    if matrix.lower() == 't':
        if cv_set.lower() == 'train':
            key = 'CV_TrainScores_t'
        else:
            key = 'CV_TestScores_t'
    else:
        if cv_set.lower() == 'train':
            key = 'CV_TrainScores_u'
        else:
            key = 'CV_TestScores_u'


    list_scores_train = [(item[0], *item[1]) for item in itertools.chain.from_iterable(model.cvParameters['PLS'][key])]
    df_scores = pandas.DataFrame(list_scores_train)
    df_scores['Group'] = df_scores[0].map(inp['mapper_sample'])
    components = [item for item in df_scores.columns if item not in [0, 'Group']]
    mapper_component = dict(zip([item for item in components], ['LV'+str(item) for item in components]))
    mapper_component[0] = 'Sample'
    df_scores.rename(columns=mapper_component, inplace = True)
    return df_scores

def scan_mv_scores_plot(df_scores, df_scree, inp, model_params, variables, var_type, data_set, n_std=1.96):
    """
    Plot scores.

    Plotting function for predictor scores.

    Parameters
    ----------
    df_scores : dataframe
        Dataframe with scores to plot.
    df_scree : dataframe
        Dataframe with scree optimization.
    inp : dict
        Method dictionary.
    model_params : dict
        Model parameters.
    variables : list
        List of dimensions.
    var_type : str
        Principial components or Latent variables.
    data_set : str
        Train or test data.
    n_std : float
        Standard deviation for Hotellings T2 ellipse.
    """
    # Create plot
    fig = plt.figure(figsize = inp['mv_figsize_score'])
    # Add plots for multiple axes
    ax1=fig.add_subplot(111, label="1", frameon = True)
    ax2=fig.add_subplot(111, label="2", frameon = False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3=fig.add_subplot(111, label="3", frameon = False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    # Plot data
    df_groups_mean = df_scores.groupby(['Sample','Group']).agg(numpy.nanmean).reset_index()
    for model in df_groups_mean['Group'].unique():
        df_group = df_groups_mean[df_groups_mean['Group'] == model].copy()
        ax1.scatter(x = df_group[variables[0]], y = df_group[variables[1]], color = inp['mapper_group_color'][model], marker = inp['mapper_group_marker'][model], edgecolor = 'k', linewidth = 0.8, label = model, s = inp['mv_labelsize']*3, alpha = 1)
        confidence_ellipse_group(df_group[variables[0]], df_group[variables[1]], ax = ax1, edgecolor = inp['mapper_group_color'][model], alpha=0.5, facecolor = inp['mapper_group_color'][model], zorder=0)

    g_ellipse = None
    X = df_groups_mean[df_groups_mean.columns[df_groups_mean.columns.isin(variables)]].values
    n_components = numpy.minimum(2, X.shape[1])
    # Compute mean and covariance
    g_ell_center = X.mean(axis=0)
    cov = numpy.cov(X, rowvar=False)
    # Width and height are "full" widths, not radius
    vals, vecs = scan_mv_sort_eigenvalues(cov)
    angle = numpy.degrees(numpy.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * numpy.sqrt(vals)
    # Compute angles of ellipse
    cos_angle = numpy.cos(numpy.radians(180. - angle))
    sin_angle = numpy.sin(numpy.radians(180. - angle))
    # Determine the elipse range
    xc = X[:, 0] - g_ell_center[0]
    yc = X[:, 1] - g_ell_center[1]
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad_cc = (xct**2 / (width / 2.)**2) + (yct**2 / (height / 2.)**2)
    # Mark the samples outside the ellipse
    outliers = rad_cc>1

    # Plot the raw points.
    g_ellipse = Ellipse(xy=g_ell_center, width=width, height=height, angle=angle, facecolor = 'none', edgecolor = 'black', linestyle = ':')
    ax1.add_patch(g_ellipse)

    # Add zero lines
    ax1.axhline(0, color = 'black', linestyle='-.', linewidth = 0.8)
    ax1.axvline(0, color = 'black', linestyle='-.', linewidth = 0.8)
    # Set label
    index_component = [int(item.lstrip('PCLV'))-1 for item in variables]
    if var_type == 'PC':
        ax1.set_xlabel(f'Scores, t [{variables[0]}]   (R$^{2}$X$_{{comp}}$ = {numpy.round(df_scree.at[index_component[0],"variance_explained_x_ratio"]*100,1)} %)', size = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_ylabel(f'Scores, t [{variables[1]}]   (R$^{2}$X$_{{comp}}$ = {numpy.round(df_scree.at[index_component[1],"variance_explained_x_ratio"]*100,1)} %)', size = inp['mv_labelsize'], fontweight = 'bold')
    if var_type == 'LV':
        ax1.set_xlabel(f'Scores, t [{variables[0]}]   (R$^{2}$X$_{{comp}}$ = {numpy.round(df_scree.at[index_component[0],"variance_explained_x_ratio"]*100,1)} %, R$^{2}$Y$_{{comp}}$ =  {numpy.round(df_scree.at[index_component[0],"variance_explained_y_ratio"]*100,1)} %)', size = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_ylabel(f'Scores, t [{variables[1]}]   (R$^{2}$X$_{{comp}}$ = {numpy.round(df_scree.at[index_component[1],"variance_explained_x_ratio"]*100,1)} %, R$^{2}$Y$_{{comp}}$ =  {numpy.round(df_scree.at[index_component[1],"variance_explained_y_ratio"]*100,1)} %)', size = inp['mv_labelsize'], fontweight = 'bold')
    
    # Scale axes
    ax1.autoscale_view()
    plt.draw()
    scaling_factor = numpy.nanmax(
        [
            numpy.nanmax(ax1.get_xlim()),
            numpy.nanmax(ax1.get_ylim()),
            numpy.nanmax(ax2.get_xlim()),
            numpy.nanmax(ax2.get_ylim()),
        ]
        )
    ax1.set_xlim(-scaling_factor,scaling_factor)
    ax1.set_ylim(-scaling_factor,scaling_factor)

    # Format
    ax1.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
    # First Legend
    legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
    # Sort both labels and handles by labels
    handles, labels = ax1.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    handles, labels = zip(*[(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]])
    leg1 = ax1.legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon = False, prop = legendproperties)

    # Second legend
    list_confidence_label = [mlines.Line2D([],[], linewidth = 0.8, linestyle = '--', color = 'black', label = r'95% CI, T$^{2}$')]
    leg2 = ax2.legend(handles = list_confidence_label, bbox_to_anchor=(0, 1.0), loc="lower left", frameon = False, prop = legendproperties)
    # Third legend
    if var_type == 'PC':
        list_score = [
            mlines.Line2D([],[], linewidth = 0, label = f'R$^{2}$X = {numpy.round(model_params["R2X"],3):.3f}')
        ]
    else:
        list_score = [
            mlines.Line2D([],[], linewidth = 0, label = f'R$^{2}$X = {numpy.round(model_params["R2X"],3):.3f}'), 
            mlines.Line2D([],[], linewidth = 0, label = f'R$^{2}$Y = {numpy.round(model_params["R2Y"],3):.3f}'),
            mlines.Line2D([],[], linewidth = 0, label = f'Q$^{2}$Y = {numpy.round(model_params["Q2Y"],3):.3f}'),
        ]

    leg3 = ax3.legend(handles = list_score, bbox_to_anchor=(1.0, 1.0), loc="lower right", frameon = False, prop = legendproperties)
    # Name append
    if var_type == 'PC':
        app = 'pca'
    else:
        app = 'pls'
    
    if inp['mv_scaling'] == True:
        app_scale = f'_{inp["mv_scaling_method"]}'
    else:
        app_scale = ''
    # Save plots
    list_bbox = [leg1, leg2, leg3]
    fig.savefig(inp['path_evaluation_mv'].joinpath(f'scores_x_{app}_{data_set}_{variables}_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
    fig.savefig(inp['path_evaluation_mv'].joinpath(f'scores_x_{app}_{data_set}_{variables}_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
    plt.close(fig)
    return

def scan_mv_sort_eigenvalues(cov):
    """
    Sort eigenvalues and eigenvectors.

    Sort eigenvalues and eigenvectors of covariance matrix.

    Parameters
    ----------
    cov : array
        Covariance matrix.

    Returns
    -------
    vals : array
        Eigenvalues.
    vecs : array
        Eigenvectors.
    """
    vals, vecs = numpy.linalg.eigh(cov)
    # vecs = vecs * numpy.sqrt(scipy.stats.chi2.ppf(0.95, n_std))
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]

def scan_mv_relevant_predictors(df_beta, df_vip, inp):
    """
    Get relevant predictors.

    Variable selection with PLS-DA model.

    Parameters
    ----------
    df_beta : dataframe
        Beta coefficient dataframe.
    df_vip : dataframe
        VIP dataframe.
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Get relevant predictor sets
    inp['predictor_relevant_beta'] = scan_mv_pls_get_relevant_variables(df_beta, inp)
    inp['predictor_relevant_vip'] = scan_mv_pls_get_relevant_variables(df_vip, inp)

    print(f'Relevant beta coefficients: {len(set(inp["predictor_relevant_beta"]["Component Name"]))}')
    print(f'Relevant VIP scores: {len(set(inp["predictor_relevant_vip"]["Component Name"]))}')
    return inp

def scan_mv_preprocessing(df):
    """
    Preprocessing for multivariate analysis.

    Creation of pivot table with missing value imputation.

    Parameters
    ----------
    df : dataframe
        Preprocessed data.

    Returns
    -------
    df_pivot : dataframe
        Preprocessed data for multivariate analysis.
    """
    # Format multivariate matrix
    df_filter = df.copy()
    df_filter = df_filter.filter(['Area','Component Name','Group']).drop_duplicates()
    df_filter['Sample'] = df_filter.groupby(['Group', 'Component Name']).cumcount()
    df_pivot = df_filter.pivot_table(index = ['Group','Sample'], columns = 'Component Name', values = 'Area').copy()
    df_pivot = df_pivot.apply(pandas.to_numeric).copy()
    df_pivot = scan_mv_preprocessing_fill_pivot_nan(df_pivot).reset_index()
    df_pivot = df_pivot[df_pivot.columns[~df_pivot.columns.isin(['Sample'])]].copy()
    return df_pivot

def scan_mv_preprocessing_fill_pivot_nan(df):
    """
    Value imputation.

    Impute missing data in pivot table.

    Parameters
    ----------
    df : dataframe
        Pivot table data with potentially missing values.

    Returns
    -------
    df : dataframe
        Pivot table data with no missing values.
    """
    df_new = pandas.DataFrame()
    for group in set(df.index.get_level_values('Group')):
        df_group = df.loc[df.index.get_level_values('Group') == group]
        for analyte in df_group.columns[~df_group.columns.isin(['Component Name'])]:
            series_fill = df_group[analyte].copy()
            # Missing at random
            series_fill[pandas.isna(series_fill)] = round(numpy.nanmean(series_fill))
            # Missing not at random
            if True in set(pandas.isna(series_fill)):
                series_fill = numpy.nanmin(df_new[analyte])/2
            df_group[analyte] = series_fill
        df_new = df_new.append(df_group)
    # Get group and analytes with all nan
    df_filled = df_new.copy()
    return df_filled

def scan_mv_preprocessing_scaling(X, inp):
    """
    Scaling.

    Apply scaling for multivariate analysis.

    Parameters
    ----------
    X : array
        Unscaled pivot table data.
    inp : dict
        Method dictionary.

    Returns
    -------
    X_new : array
        Scaled pivot table data.
    """
    if inp['mv_scaling'] == True:
        scaler = scan_mv_preprocessing_create_scaler(inp)
        X_new = scaler.fit_transform(X)
    else:
        X_new = X.copy()
    return X_new

def scan_mv_preprocessing_create_scaler(inp):
    """
    Create scaler.

    Create user specified scaler for multivariate analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    scaler : object
        Scaler object.
    """
    if inp['mv_scaling'] == True:
        # Scaling
        if inp['mv_scaling_method'].lower() == 'auto':
            scaler = CustomScalerAuto()
        elif inp['mv_scaling_method'].lower() == 'range':
            scaler = CustomScalerRange()
        elif inp['mv_scaling_method'].lower() == 'pareto':
            scaler = CustomScalerPareto()
        elif inp['mv_scaling_method'].lower() == 'vast':
            scaler = CustomScalerVast()
        elif inp['mv_scaling_method'].lower() == 'level':
            scaler = CustomScalerLevel()
        else:
            scaler = CustomScalerAuto(with_mean = False, with_std = False)
    else:
        scaler = CustomScalerAuto(with_mean = False, with_std = False)
    return scaler

def scan_mv_preprocessing_create_cv(inp):
    """
    Create cross-validation object.

    Create user specified cross-validation object for multivariate analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    cv : object
        Cross-validation object.
    """
    if inp['mv_cv_iterator'].lower() == 'kfold':
        if inp['mv_cv_kfold'] < 2:
            inp['mv_cv_kfold'] = 2

        if inp['mv_cv_stratified'] == True:
            if inp['mv_cv_repeated'] == True:
                cv = model_selection.RepeatedStratifiedKFold(n_splits = inp['mv_cv_kfold'], n_repeats = inp['mv_cv_repetitions'])
            else:
                cv = model_selection.StratifiedKFold(n_splits = inp['mv_cv_kfold'], shuffle=True)
        else:
            if inp['mv_cv_repeated'] == True:
                cv = model_selection.RepeatedKFold(n_splits = inp['mv_cv_kfold'], n_repeats = inp['mv_cv_repetitions'])
            else:
                cv = model_selection.KFold(n_splits = inp['mv_cv_kfold'], shuffle=True)
    else:
        raise ValueError('Check iterator abbreviation.')
    return cv

def scan_mv_optimization(model_type, X, y_group, inp):
    """
    Hyperparameter optimization.

    Identify number of principial components or latent variables for multivariate modelling.

    Parameters
    ----------
    model_type : str
        Model type.
    X : array
        Pivot data.
    y_group : array
        Group labels.
    inp : dict
        Method dictionary.

    Returns
    -------
    df_scree : dataframe
        Scree dataframe.
    """
    # Get maximum number of potential latent variables
    if X.shape[0] <= X.shape[1]:
        max_components = X.shape[0] - 1
    else:
        max_components = X.shape[1]
    # Cap maximum number of potential latent variables
    if max_components > 12:
        max_components = 12

    if model_type.lower() == 'pca':
        df_scree = pandas.DataFrame([f'PC{item}' for item in numpy.array(range(1, max_components+1)).T], columns = ['component'])
        models = []
        cv = scan_mv_preprocessing_create_cv(inp)
        scaler = scan_mv_preprocessing_create_scaler(inp)
        for i in range(max_components):
            model_i = ChemometricsPCA(ncomps=i+1, scaler = scaler)
            model_i.fit(x = X)
            model_i.cross_validation(x = X, y = y_group, cv_method = cv)
            models.append(model_i)
        
        df_scree['R2X'] = [x.modelParameters['R2X'] for x in models]
        df_scree['Q2X'] = [x.cvParameters['Q2X'] for x in models]
    else:
        df_scree = pandas.DataFrame([f'LV{item}' for item in numpy.array(range(1, max_components+1)).T], columns = ['component'])
        models = []
        cv = scan_mv_preprocessing_create_cv(inp)
        scaler = scan_mv_preprocessing_create_scaler(inp)
        for i in range(max_components):
            model_i = ChemometricsPLSDA(ncomps=i+1, xscaler = scaler)
            model_i.fit(x = X, y = y_group)
            model_i.cross_validation(x = X, y = y_group, cv_method = cv, outputdist = True)
            models.append(model_i)

        df_scree['R2X'] = [x.modelParameters['PLS']['R2X'] for x in models]
        df_scree['R2Y'] = [x.modelParameters['PLS']['R2Y'] for x in models]
        df_scree['Q2Y'] = [x.cvParameters['PLS']['Q2Y'] for x in models]
    return df_scree

def scan_mv_optimization_min(df_scree_all, model_type, inp):
    """
    Identify hyperparameter.

    Identify minimal nessesary number of principial components or latent variables for multivariate modelling.

    Parameters
    ----------
    df_scree_all : dataframe
        Scree dataframe.
    model_type : str
        Model type.

    Returns
    -------
    df_scree : dataframe
        Scree dataframe.
    n_components : int
        Number of components.
    """
    df_scree = df_scree_all.copy()
    # Increase less than
    cutoff = 0.05
    if model_type.lower() == 'pca':
        kpi = 'Q2X'
    else:
        kpi = 'Q2Y'

    percent_cutoff = numpy.where(abs(numpy.diff(df_scree[kpi])) / abs(df_scree[kpi][0:-1]) < cutoff)[0]
    if len(percent_cutoff) > 0:
        n_components = numpy.nanmin(percent_cutoff)+1
    else:
        n_components = numpy.where(numpy.nanmax(df_scree[kpi]))[0]+1

    if n_components < 3:
        n_components = 3
    df_scree.at[n_components-1,'index_selection'] = True
    return df_scree, n_components

def scan_mv_optimization_scree_plot(df_scree, model_type, inp):
    """
    Plot scree.

    Plotting function for hyperparameter identification.

    Parameters
    ----------
    df_scree : dataframe
        Dataframe with scree optimization.
    model_type : str
        Model type.
    inp : dict
        Method dictionary.
    """
    # Scree plot PCA
    fig, ax1 = plt.subplots(figsize = inp['mv_figsize_scree'])
    df_plot = df_scree.copy()
    x = range(1,len(df_scree['component'])+1)
    palette = seaborn.color_palette('Set1', n_colors=7)
    list_bbox = []
    if model_type.lower() == 'pca':
        app = 'pca'
        ax1.plot(x, df_scree['R2X'], color = palette[0], marker = 'o', linestyle='-', label = 'R${^2}$X')
        ax1.plot(x, df_scree['Q2X'], color = palette[1],  marker = 'o', linestyle='-', label = 'Q${^2}$X')
        ax1.axvline(df_plot[df_plot['index_selection'] == True].index[0]+1, linewidth = 0.8, color = 'red')
    else:
        app = 'pls'
        ax1.plot(x, df_plot['R2X'], color = palette[0], marker = 'o', linestyle='-', label = 'R${^2}$X')
        ax1.plot(x, df_plot['R2Y'], color = palette[2],  marker = 'o', linestyle='-', label = 'R${^2}$Y')
        ax1.plot(x, df_plot['Q2Y'], color = palette[3],  marker = 'o', linestyle='-', label = 'Q${^2}$Y')
        ax1.axvline(df_plot[df_plot['index_selection']==True].index[0]+1, linewidth = 0.8, color = 'red')
    
    ax1.set_xticks(x)
    plt.draw()
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation = '0')
    ax1.set_xlabel(f'Number of components', size = inp['mv_labelsize'], fontweight = 'bold')
    ax1.set_ylabel('R$^{2}$, Q$^{2}$', size = inp['mv_labelsize'], fontweight = 'bold')
    ax1.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
    list_bbox.append(ax1.legend(frameon = False, loc = 'lower left', bbox_to_anchor = (0,1), bbox_transform = ax1.transAxes))

    # Name append    
    if inp['mv_scaling'] == True:
        app_scale = f'_{inp["mv_scaling_method"]}'
    else:
        app_scale = ''

    fig.savefig(inp['path_evaluation_mv'].joinpath(f'scree_{app}_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_extra_artists = list_bbox, bbox_inches = 'tight', dpi = 800)
    fig.savefig(inp['path_evaluation_mv'].joinpath(f'scree_{app}_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_extra_artists = list_bbox, bbox_inches = 'tight', format = 'svg')
    plt.close(fig)
    return

def scan_mv_pca_calculation_cv(X, y_group, n_components, inp):
    """
    PCA modelling.

    Model PCA with scaling and cross-validation.

    Parameters
    ----------
    X : array
        Pivot data.
    y_group : array
        Group labels.
    n_components : int
        Number of components.
    inp : dict
        Method dictionary.

    Returns
    -------
    pca : object
        PCA object.
    """
    scaler = scan_mv_preprocessing_create_scaler(inp)
    cv = scan_mv_preprocessing_create_cv(inp)
    pca = ChemometricsPCA(ncomps=n_components, scaler = scaler)
    pca.fit(x = X)
    pca.cross_validation(x = X, y = y_group, cv_method = cv, outputdist = True)
    return pca

def scan_mv_pls_calculation_cv(X, y_group, n_components, inp):
    """
    PLS-DA modelling.

    Model PLS-DA with scaling and cross-validation.

    Parameters
    ----------
    X : array
        Pivot data.
    y_group : array
        Group labels.
    n_components : int
        Number of components.
    inp : dict
        Method dictionary.

    Returns
    -------
    plsda : object
        PLS-DA object.
    """
    # Run PLS with suggested number of components
    scaler = scan_mv_preprocessing_create_scaler(inp)
    cv = scan_mv_preprocessing_create_cv(inp)
    plsda = ChemometricsPLSDA(ncomps=n_components, xscaler = scaler)
    plsda.fit(x = X, y = y_group)
    plsda.cross_validation(x = X, y = y_group, cv_method = cv, outputdist = True)
    plsda.permutation_test(x = X, y = y_group, nperms = 100, cv_method = cv, outputdist = True)
    plsda.bootstrap_test(x = X, y = y_group, nboots = 100, stratify = True, outputdist = True)

    return plsda

def scan_mv_pca_model_params(model, df_scree):
    """
    Get model params.

    Extract model params of PCA.

    Parameters
    ----------
    model : object
        Fitted PCA object.

    Returns
    -------
    model_params : dict
        Model parameter.
    """
    # Select parameters
    n_components = model._ncomps
    model_params = model.modelParameters
    model_params['Q2X'] = model.cvParameters['Q2X']
    df_scree['variance_explained_x_ratio'] = df_scree['R2X'].diff().fillna(df_scree['R2X'].iloc[0])
    df_scree['variance_explained_x_ratio_cumsum'] = df_scree['variance_explained_x_ratio'].cumsum()

    return model_params

def scan_mv_pls_model_params(model, df_scree):
    """
    Get model params.

    Extract model params of PLS-DA.

    Parameters
    ----------
    model : object
        Fitted PLS-DA object.

    Returns
    -------
    model_params : dict
        Model parameter.
    """
    # Select parameters
    n_components = model._ncomps
    model_params = model.modelParameters['PLS']
    model_params['Q2Y'] = model.cvParameters['PLS']['Q2Y']

    df_scree['variance_explained_x_ratio'] = df_scree['R2X'].diff().fillna(df_scree['R2X'].iloc[0])
    df_scree['variance_explained_y_ratio'] = df_scree['R2Y'].diff().fillna(df_scree['R2Y'].iloc[0])
    df_scree['variance_explained_x_ratio_cumsum'] = df_scree['variance_explained_x_ratio'].cumsum()
    df_scree['variance_explained_y_ratio_cumsum'] = df_scree['variance_explained_y_ratio'].cumsum()
    
    return model_params

def scan_mv_covariance_matrix(X):
    """
    Get covariance matrix.

    Create covariance matrix, eigenvalues and eigenvectors.

    Parameters
    ----------
    X : array
        Pivot data.

    Returns
    -------
    cov : array
        Covariance matrix.
    eig_val : array
        Eigenvalues.
    eig_vec : array
        Eigenvectors.
    """
    # Covariance matrix
    cov = numpy.cov(X.T)
    # Get eigenvalues and eigenvectors with scipy and consider non-normal matrices
    eig_val, eig_vec = scipy.linalg.eigh(cov)
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]
    return cov, eig_val, eig_vec

def scan_mv_hotelling_T2_ellipse(df_scores, variables, alpha = 0.05):
    """
    Get Hotelling T2 parameter.

    Create covariance matrix, eigenvalues and eigenvectors.

    Parameters
    ----------
    df_scores : dataframe
        Dataframe with scores.
    variables : list
        List with model components.
    alpha : float
        Error of probability.

    Returns
    -------
    hotelling_t2 : array
        Hotelling T2 parameter.
    """
    # For PCA and PLS
    # Get scores array
    scores = numpy.array(df_scores.values)
    # Get index of compounds
    comps = [int(item.lstrip('PCLV'))-1 for item in variables]
    # Allocate samples
    n_samples = scores.shape[0]
    n_components = len(comps)
    ellips = scores[:, comps] ** 2
    ellips = 1/ n_samples*(ellips.sum(0))
    # F statistic
    a = (n_samples - 1) / n_samples * n_components * (n_samples ** 2 - 1) / (n_samples * (n_samples - n_components))
    a = a * scipy.stats.f.ppf(1-alpha, n_components, n_samples - n_components)
    # Append
    hotelling_t2 = []
    for comp in range(n_components):
        hotelling_t2.append(numpy.sqrt((a * ellips[comp])))
    return numpy.array(hotelling_t2)

def scan_mv_pls_beta(model, labels_group, labels_variable):
    """
    Get beta coefficients.

    Extract beta coefficients of PLS-DA model.

    Parameters
    ----------
    model : object
        Fitted PLS-DA object.
    labels_group : list
        Group labels.
    labels_variable : list
        Variable labels.

    Returns
    -------
    df_final : dataframe
        Dataframe with beta coefficients.    
    """
    array_mean = numpy.nanmean(model.bootstrapParameters['PLS']['boot_Beta'], axis = 0)
    array_sem = scipy.stats.sem(model.bootstrapParameters['PLS']['boot_Beta'])
    array_ci_lower = numpy.nanpercentile(model.bootstrapParameters['PLS']['boot_Beta'], q = 2.5, axis = 0, interpolation = 'linear')
    array_ci_upper = numpy.nanpercentile(model.bootstrapParameters['PLS']['boot_Beta'], q = 97.5, axis = 0, interpolation = 'linear')

    if len(labels_group) > 2:
        df_mean = pandas.DataFrame(array_mean.T, columns = [items+'_mean' for items in labels_group])
        df_sem = pandas.DataFrame(array_sem.T, columns = [items+'_std' for items in labels_group])
        df_ci_lower = pandas.DataFrame(array_ci_lower.T, columns = [items+'_ci_lower' for items in labels_group])
        df_ci_upper = pandas.DataFrame(array_ci_upper.T, columns = [items+'_ci_upper' for items in labels_group])

        df_final = pandas.concat([df_mean, df_sem, df_ci_lower, df_ci_upper], axis = 1)
        df_final['Component Name'] = labels_variable
        
        for i, row in df_final.iterrows():
            for label in labels_group:
                df_final.at[i, f'{label}_interval'] = pandas.Interval(left = df_final.at[i, f'{label}_ci_lower'], right = df_final.at[i, f'{label}_ci_upper'], closed = 'both')
                df_final.at[i, f'{label}_relevant'] = (0 not in df_final.at[i, f'{label}_interval'])
    else:
        df_mean = pandas.DataFrame(array_mean.T, columns = ['mean'])
        df_sem = pandas.DataFrame(array_sem.T, columns = ['std'])
        df_ci_lower = pandas.DataFrame(array_ci_lower.T, columns = ['ci_lower'])
        df_ci_upper = pandas.DataFrame(array_ci_upper.T, columns = ['ci_upper'])

        df_final = pandas.concat([df_mean, df_sem, df_ci_lower, df_ci_upper], axis = 1)
        df_final['Component Name'] = labels_variable
        
        for i, row in df_final.iterrows():
            df_final.at[i, f'interval'] = pandas.Interval(left = df_final.at[i, f'ci_lower'], right = df_final.at[i, f'ci_upper'], closed = 'both')
            df_final.at[i, f'relevant'] = (0 not in df_final.at[i, f'interval'])
    return df_final

def scan_mv_pls_vip(model, labels_group, labels_variable):
    """
    Get VIP scores.

    Extract VIP of PLS-DA model.

    Parameters
    ----------
    model : object
        Fitted PLS-DA object.
    labels_group : list
        Group labels.
    labels_variable : list
        Variable labels.

    Returns
    -------
    df_final : dataframe
        Dataframe with VIP scores.    
    """
    array_mean = numpy.nanmean(model.bootstrapParameters['PLS']['boot_VIPw'], axis = 0)
    array_sem = scipy.stats.sem(model.bootstrapParameters['PLS']['boot_VIPw'])
    array_ci_lower = numpy.nanpercentile(model.bootstrapParameters['PLS']['boot_VIPw'], q = 2.5, axis = 0, interpolation = 'linear')
    array_ci_upper = numpy.nanpercentile(model.bootstrapParameters['PLS']['boot_VIPw'], q = 97.5, axis = 0, interpolation = 'linear')
    
    if len(labels_group) > 2:
        df_mean = pandas.DataFrame(array_mean.T, columns = [items+'_mean' for items in labels_group])
        df_sem = pandas.DataFrame(array_sem.T, columns = [items+'_std' for items in labels_group])
        df_ci_lower = pandas.DataFrame(array_ci_lower.T, columns = [items+'_ci_lower' for items in labels_group])
        df_ci_upper = pandas.DataFrame(array_ci_upper.T, columns = [items+'_ci_upper' for items in labels_group])

        df_final = pandas.concat([df_mean, df_sem, df_ci_lower, df_ci_upper], axis = 1)
        df_final['Component Name'] = labels_variable
        
        for label in labels_group:
            df_final[f'{label}_relevant'] = (df_final[f'{label}_ci_lower']>=1)
    else:
        df_mean = pandas.DataFrame(array_mean.T, columns = ['mean'])
        df_sem = pandas.DataFrame(array_sem.T, columns = ['std'])
        df_ci_lower = pandas.DataFrame(array_ci_lower.T, columns = ['ci_lower'])
        df_ci_upper = pandas.DataFrame(array_ci_upper.T, columns = ['ci_upper'])

        df_final = pandas.concat([df_mean, df_sem, df_ci_lower, df_ci_upper], axis = 1)
        df_final['Component Name'] = labels_variable
        
        df_final[f'relevant'] = (df_final[f'ci_lower']>=1)

    return df_final

def scan_mv_pls_rocauc(model, labels_group):
    """
    Get ROC and AUC.

    Extract Receiver-Operator-Characteristic and AUROC of PLS-DA model.

    Parameters
    ----------
    model : object
        Fitted PLS-DA object.
    labels_group : list
        Group labels.
    labels_variable : list
        Variable labels.

    Returns
    -------
    dict_rocauc : dict
        Dictionary with ROC and AUC.    
    """
    # Preallocate
    dict_rocauc = {}
        
    # Account for two classes
    if len(labels_group) == 2:
        labels_group = ["0"]

    # Train
    array_train_fpr_mean = numpy.nanmean(model.bootstrapParameters['DA']['boot_TrainROC_fpr'], axis = 0)
    array_train_tpr_mean = numpy.nanmean(model.bootstrapParameters['DA']['boot_TrainROC_tpr'], axis = 0)
    array_train_auc_mean = numpy.nanmean(model.bootstrapParameters['DA']['boot_TrainAUC'], axis = 0)

    array_train_tpr_std = numpy.nanstd(model.bootstrapParameters['DA']['boot_TrainROC_tpr'], axis = 0)
    array_train_auc_std = numpy.nanstd(model.bootstrapParameters['DA']['boot_TrainAUC'], axis = 0)

    array_train_tpr_ci_lower = numpy.nanpercentile(model.bootstrapParameters['DA']['boot_TrainROC_tpr'], q = 2.5, axis = 0)
    array_train_tpr_ci_upper = numpy.nanpercentile(model.bootstrapParameters['DA']['boot_TrainROC_tpr'], q = 97.5, axis = 0)
    
    df_train_fpr_mean = pandas.DataFrame(array_train_fpr_mean, columns = [item+'_fpr_mean' for item in labels_group])
    df_train_tpr_mean = pandas.DataFrame(array_train_tpr_mean, columns = [item+'_tpr_mean' for item in labels_group])
    df_train_auc_mean = pandas.DataFrame([array_train_auc_mean], columns = [item+'_auc_mean' for item in labels_group])

    df_train_tpr_std = pandas.DataFrame(array_train_tpr_std, columns = [item+'_tpr_std' for item in labels_group])
    df_train_auc_std = pandas.DataFrame([array_train_auc_std], columns = [item+'_auc_std' for item in labels_group])

    df_train_tpr_ci_lower = pandas.DataFrame(array_train_tpr_ci_lower, columns = [item+'_tpr_ci_lower' for item in labels_group])
    df_train_tpr_ci_upper = pandas.DataFrame(array_train_tpr_ci_upper, columns = [item+'_tpr_ci_upper' for item in labels_group])

    dict_rocauc['TrainROC'] = pandas.concat([df_train_fpr_mean, df_train_tpr_mean, df_train_tpr_std, df_train_tpr_ci_lower, df_train_tpr_ci_upper], axis = 1)
    dict_rocauc['TrainAUC'] = pandas.concat([df_train_auc_mean, df_train_auc_std], axis = 1)

    # Test
    array_test_fpr_mean = numpy.nanmean(model.bootstrapParameters['DA']['boot_TestROC_fpr'], axis = 0)
    array_test_tpr_mean = numpy.nanmean(model.bootstrapParameters['DA']['boot_TestROC_tpr'], axis = 0)
    array_test_auc_mean = numpy.nanmean(model.bootstrapParameters['DA']['boot_TestAUC'], axis = 0)

    array_test_tpr_std = numpy.nanstd(model.bootstrapParameters['DA']['boot_TestROC_tpr'], axis = 0)
    array_test_auc_std = numpy.nanstd(model.bootstrapParameters['DA']['boot_TestAUC'], axis = 0)

    array_test_tpr_ci_lower = numpy.nanpercentile(model.bootstrapParameters['DA']['boot_TestROC_tpr'], q = 2.5, axis = 0)
    array_test_tpr_ci_upper = numpy.nanpercentile(model.bootstrapParameters['DA']['boot_TestROC_tpr'], q = 97.5, axis = 0)
    
    df_test_fpr_mean = pandas.DataFrame(array_test_fpr_mean, columns = [item+'_fpr_mean' for item in labels_group])
    df_test_tpr_mean = pandas.DataFrame(array_test_tpr_mean, columns = [item+'_tpr_mean' for item in labels_group])
    df_test_auc_mean = pandas.DataFrame([array_test_auc_mean], columns = [item+'_auc_mean' for item in labels_group])

    df_test_tpr_std = pandas.DataFrame(array_test_tpr_std, columns = [item+'_tpr_std' for item in labels_group])
    df_test_auc_std = pandas.DataFrame([array_test_auc_std], columns = [item+'_auc_std' for item in labels_group])

    df_test_tpr_ci_lower = pandas.DataFrame(array_test_tpr_ci_lower, columns = [item+'_tpr_ci_lower' for item in labels_group])
    df_test_tpr_ci_upper = pandas.DataFrame(array_test_tpr_ci_upper, columns = [item+'_tpr_ci_upper' for item in labels_group])

    dict_rocauc['TestROC'] = pandas.concat([df_test_fpr_mean, df_test_tpr_mean, df_test_tpr_std, df_test_tpr_ci_lower, df_test_tpr_ci_upper], axis = 1)
    dict_rocauc['TestAUC'] = pandas.concat([df_test_auc_mean, df_test_auc_std], axis = 1)
    return dict_rocauc

def scan_mv_pls_get_relevant_variables(df, inp):
    """
    Get relevant predictors.

    Variable selection with PLS-DA model subfunction.

    Parameters
    ----------
    df_beta : dataframe
        Beta coefficient dataframe.
    df_vip : dataframe
        VIP dataframe.
    inp : dict
        Method dictionary.

    Returns
    -------
    set_relevant : set
        Relevant predictors.
    """
    df_relevant = pandas.DataFrame()
    labels_group = pandas.Series([item.split('_')[0] for item in df.columns if item not in ['Component Name']]).unique()

    if set(inp['groups']).issubset(labels_group):
        for group in labels_group:
            df_group = df[df[group+'_relevant'] == True].copy()
            df_group['group'] = group
            df_relevant = df_relevant.append(df_group)
    else:
        df_group = df[df['relevant'] == True].copy()
        df_group['group'] = 'both'
        df_relevant = df_relevant.append(df_group)
    df_relevant = df_relevant[df_relevant.columns[df_relevant.columns.isin(['Component Name','group'])]].copy()
    return df_relevant

def scan_mv_pls_vip_plot_full(df_vip, labels_group, inp):
    """
    Plot VIP.

    Plotting function for VIP scores.

    Parameters
    ----------
    df_vip : dataframe
        Dataframe with VIPs to plot.
    labels_group : list
        Group labels.
    inp : dict
        Method dictionary.
    """
    if len(labels_group) > 2:
        for item in labels_group:
            df_hold = df_vip.copy().filter(['Component Name', item+'_mean', item+'_ci_lower', item+'_ci_upper'])
            
            df_hold = df_hold.copy().sort_values(by = [item+'_mean'], ascending = True).reset_index(drop=True)
            x1 = len(df_hold)
            df_hold = df_hold.loc[len(df_hold)-inp['mv_vip_number']:].copy()
            x2 = len(df_hold)

            # Create plot   
            fig = plt.figure(figsize = inp['mv_figsize_vip'])
            # Add plots for multiple axes
            ax1=fig.add_subplot(111, label="1", frameon = True)
            ax2=fig.add_subplot(111, label="2", frameon = False)
            ax2.get_xaxis().set_visible(False)
            ax2.yaxis.tick_right()
            #ax2.get_yaxis().set_visible(False)     
            
            # Plot data
            xerr = [df_hold[item+'_mean']-df_hold[item+'_ci_lower'], df_hold[item+'_ci_upper']-df_hold[item+'_mean']]
            markersize = 4
            points = ax1.errorbar(x = df_hold[item+'_mean'], y = df_hold.index, xerr=xerr, linewidth = 0.8, fmt = 'o', markersize = markersize, elinewidth=1, capsize=markersize, label = item)
            if numpy.nanmin(df_hold[item+'_mean'])<1:
                ax1.axvline(1, linewidth = 0.8, color = 'red')

            # Set grid
            ax1.set_axisbelow(True)
            ax1.yaxis.grid()        

            # Scale axis
            ax1.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)
            ax2.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)

            plt.locator_params(axis='x', nbins=5)
            ax1.yaxis.set_ticks(df_hold.index)
            ax2.yaxis.set_ticks(df_hold.index)
            ax1.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
            ax2.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
            ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
            ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])

            plt.draw()

            # Set standard label
            labels1 = [label for label in df_hold['Component Name']]
            ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = inp['mv_labelsize'], fontweight = 'bold')
            ax1.set_yticklabels(labels1, fontsize = inp['mv_labelsize'], va = 'center', ha = 'right', fontweight = 'bold')
            
            # Set full label
            if inp['mv_label_vip'] == True:
                labels2 = [inp['mapper_name'][label] for label in df_hold['Component Name']]
                ax2.set_yticklabels(labels2, fontsize = inp['mv_labelsize'], va = 'center', ha = 'left', fontweight = 'bold')

            # Set label
            ax1.set_xlabel(f'VIP score', fontsize = inp['mv_labelsize'], fontweight = 'bold')
            ax1.set_ylabel(None)

            # Add non displayed data information
            legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
            list_bbox = []

            handles_class = mlines.Line2D([], [], marker=None, markersize=0, label=f'{item}')   
            leg1 = ax1.legend(handles = [handles_class], handlelength = False, bbox_to_anchor=(0, 1), loc="lower left", frameon = False, prop = legendproperties)
            list_bbox.append(leg1)

            if x1-x2 > 0:
                handles_annotation = mlines.Line2D([], [], marker=None,markersize=0, label=f'VIP Top {inp["mv_vip_number"]}')
                leg2 = ax2.legend(handles = [handles_annotation], handlelength = False, bbox_to_anchor=(1, 1), loc="lower right", frameon = False, prop = legendproperties)
                list_bbox.append(leg2)
            
            # Name append       
            if inp['mv_scaling'] == True:
                app_scale = f'_{inp["mv_scaling_method"]}'
            else:
                app_scale = ''
            fig.savefig(inp['path_evaluation_mv'].joinpath(f'vip_pls_{item}_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
            fig.savefig(inp['path_evaluation_mv'].joinpath(f'vip_pls_{item}_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
            plt.close(fig)
    else:
        df_hold = df_vip.copy().filter(['Component Name', 'mean', 'ci_lower', 'ci_upper'])
            
        df_hold = df_hold.copy().sort_values(by = ['mean'], ascending = True).reset_index(drop=True)
        x1 = len(df_hold)
        df_hold = df_hold.loc[len(df_hold)-inp['mv_vip_number']:].copy()
        x2 = len(df_hold)

        # Create plot   
        fig = plt.figure(figsize = inp['mv_figsize_vip'])
        # Add plots for multiple axes
        ax1=fig.add_subplot(111, label="1", frameon = True)
        ax2=fig.add_subplot(111, label="2", frameon = False)
        ax2.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        #ax2.get_yaxis().set_visible(False)     
        
        # Plot data
        xerr = [df_hold['mean']-df_hold['ci_lower'], df_hold['ci_upper']-df_hold['mean']]
        markersize = 4
        points = ax1.errorbar(x = df_hold['mean'], y = df_hold.index, xerr=xerr, linewidth = 0.8, fmt = 'o', markersize = markersize, elinewidth=1, capsize=markersize)
        if numpy.nanmin(df_hold['mean'])<1:
            ax1.axvline(1, linewidth = 0.8, color = 'red')

        # Set grid
        ax1.set_axisbelow(True)
        ax1.yaxis.grid()        

        # Scale axis
        ax1.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)
        ax2.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)

        plt.locator_params(axis='x', nbins=5)
        ax1.yaxis.set_ticks(df_hold.index)
        ax2.yaxis.set_ticks(df_hold.index)
        ax1.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])

        plt.draw()

        # Set standard label
        labels1 = [label for label in df_hold['Component Name']]
        ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_yticklabels(labels1, fontsize = inp['mv_labelsize'], va = 'center', ha = 'right', fontweight = 'bold')
        
        # Set full label
        if inp['mv_label_vip'] == True:
            labels2 = [inp['mapper_name'][label] for label in df_hold['Component Name']]
            ax2.set_yticklabels(labels2, fontsize = inp['mv_labelsize'], va = 'center', ha = 'left', fontweight = 'bold')

        # Set label
        ax1.set_xlabel(f'VIP score', fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_ylabel(None)

        # Add non displayed data information
        legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
        list_bbox = []

        if x1-x2 > 0:
            handles_annotation = mlines.Line2D([], [], marker=None,markersize=0, label=f'VIP Top {inp["mv_vip_number"]}')
            leg2 = ax2.legend(handles = [handles_annotation], handlelength = False, bbox_to_anchor=(1, 1), loc="lower right", frameon = False, prop = legendproperties)
            list_bbox.append(leg2)

        # Name append       
        if inp['mv_scaling'] == True:
            app_scale = f'_{inp["mv_scaling_method"]}'
        else:
            app_scale = ''
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'vip_pls_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'vip_pls_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
        plt.close(fig)
    return

def scan_mv_pls_vip_plot_relevant(df_vip, labels_group, inp):
    """
    Plot VIP.

    Plotting function for user-specified VIP scores.

    Parameters
    ----------
    df_vip : dataframe
        Dataframe with VIPs to plot.
    labels_group : list
        Group labels.
    inp : dict
        Method dictionary.
    """
    if len(labels_group) > 2:
        # Create plot   
        fig = plt.figure(figsize = inp['mv_figsize_vip'])
        # Add plots for multiple axes
        ax1=fig.add_subplot(111, label="1", frameon = True)
        ax2=fig.add_subplot(111, label="2", frameon = False)
        ax2.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        #ax2.get_yaxis().set_visible(False)   

        mapper_sorting = dict(zip(inp['pre_list_relevant'],range(len(inp['pre_list_relevant']))))
        df_vip = df_vip[df_vip['Component Name'].isin(set(inp['pre_list_relevant']))].copy().reset_index(drop=True)
        df_vip['sorter'] = df_vip['Component Name'].map(mapper_sorting)
        df_vip = df_vip.copy().sort_values(by = ['sorter'], ascending = False).reset_index(drop=True)
        df_vip = df_vip.copy().filter([item for item in df_vip.columns if item != 'sorter'])

        for item in labels_group:
            df_hold = df_vip.filter(['Component Name', item+'_mean', item+'_ci_lower', item+'_ci_upper', item+'_relevant']).copy()
            
            color = inp['mapper_group_color'][item]
            markersize = 3

            df_rel = df_hold[df_hold[item+'_relevant'] == True].copy()
            xerr_rel = [df_rel[item+'_mean']-df_rel[item+'_ci_lower'], df_rel[item+'_ci_upper']-df_rel[item+'_mean']]
            ax1.errorbar(x = df_rel[item+'_mean'], y = df_rel.index, xerr=xerr_rel, linewidth = 0.8, color = color, fmt = 'o', ecolor=color, markersize = markersize, elinewidth=1, capsize=markersize, label = item+r' (VIP > 1)')
            
            df_notrel = df_hold[df_hold[item+'_relevant'] == False].copy()
            xerr_notrel = [df_notrel[item+'_mean']-df_notrel[item+'_ci_lower'], df_notrel[item+'_ci_upper']-df_notrel[item+'_mean']]
            ax1.errorbar(x = df_notrel[item+'_mean'], y = df_notrel.index, xerr=xerr_notrel, linewidth = 0.8, color = 'grey', fmt = 'o', ecolor='grey', markersize = markersize, elinewidth=1, capsize=markersize, label = item+r' (VIP $\leq$ 1)')

        if numpy.nanmin(df_hold[item+'_mean'])<1:
            ax1.axvline(1, linewidth = 0.8, color = 'red')

        # Set grid
        ax1.set_axisbelow(True)
        ax1.yaxis.grid()        

        # Scale axis
        ax1.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)
        ax2.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)

        plt.locator_params(axis='x', nbins=5)
        ax1.yaxis.set_ticks(df_hold.index)
        ax2.yaxis.set_ticks(df_hold.index)
        ax1.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])

        plt.draw()

        # Set standard label
        #labels1 = [label for label in df_hold['Component Name']]
        ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_yticklabels([None]*len(ax1.get_yticklabels()), fontsize = inp['mv_labelsize'], va = 'center', ha = 'right', fontweight = 'bold')
        
        # Set full label
        if inp['mv_label_vip'] == True:
            labels2 = [inp['mapper_name'][label] for label in df_hold['Component Name']]
        else:
            labels2 = [label for label in df_hold['Component Name']]

        ax2.set_yticklabels(labels2, fontsize = inp['mv_labelsize'], va = 'center', ha = 'left', fontweight = 'bold')
        # Set label
        ax1.set_xlabel(f'VIP score', fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_ylabel(None)

        # Legend
        legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
        # Sort both labels and handles by labels
        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        leg = ax1.legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc="lower right", frameon = False, prop = legendproperties)
        
        # Name append    
        if inp['mv_scaling'] == True:
            app_scale = f'_{inp["mv_scaling_method"]}'
        else:
            app_scale = ''

        list_bbox = [leg]
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'vip_pls_SN{inp["pre_signal_noise"]}_relevant{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'vip_pls_SN{inp["pre_signal_noise"]}_relevant{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
        plt.close(fig)
    else:
        # Create plot   
        fig = plt.figure(figsize = inp['mv_figsize_vip'])
        # Add plots for multiple axes
        ax1=fig.add_subplot(111, label="1", frameon = True)
        ax2=fig.add_subplot(111, label="2", frameon = False)
        ax2.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        #ax2.get_yaxis().set_visible(False)   

        mapper_sorting = dict(zip(inp['pre_list_relevant'],range(len(inp['pre_list_relevant']))))
        df_vip = df_vip[df_vip['Component Name'].isin(set(inp['pre_list_relevant']))].copy().reset_index(drop=True)
        df_vip['sorter'] = df_vip['Component Name'].map(mapper_sorting)
        df_vip = df_vip.copy().sort_values(by = ['sorter'], ascending = False).reset_index(drop=True)
        df_vip = df_vip.copy().filter([item for item in df_vip.columns if item != 'sorter'])

        df_hold = df_vip.filter(['Component Name', 'mean', 'ci_lower', 'ci_upper', 'relevant']).copy()
        
        color = 'black'
        markersize = 3

        df_rel = df_hold[df_hold['relevant'] == True].copy()
        xerr_rel = [df_rel['mean']-df_rel['ci_lower'], df_rel['ci_upper']-df_rel['mean']]
        ax1.errorbar(x = df_rel['mean'], y = df_rel.index, xerr=xerr_rel, linewidth = 0.8, color = color, fmt = 'o', ecolor=color, markersize = markersize, elinewidth=1, capsize=markersize, label = 'VIP > 1')
        
        df_notrel = df_hold[df_hold['relevant'] == False].copy()
        xerr_notrel = [df_notrel['mean']-df_notrel['ci_lower'], df_notrel['ci_upper']-df_notrel['mean']]
        ax1.errorbar(x = df_notrel['mean'], y = df_notrel.index, xerr=xerr_notrel, linewidth = 0.8, color = 'grey', fmt = 'o', ecolor='grey', markersize = markersize, elinewidth=1, capsize=markersize, label = 'VIP $\leq$ 1')

        if numpy.nanmin(df_hold['mean'])<1:
            ax1.axvline(1, linewidth = 0.8, color = 'red')

        # Set grid
        ax1.set_axisbelow(True)
        ax1.yaxis.grid()        

        # Scale axis
        ax1.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)
        ax2.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)

        plt.locator_params(axis='x', nbins=5)
        ax1.yaxis.set_ticks(df_hold.index)
        ax2.yaxis.set_ticks(df_hold.index)
        ax1.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])

        plt.draw()

        # Set standard label
        #labels1 = [label for label in df_hold['Component Name']]
        ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_yticklabels([None]*len(ax1.get_yticklabels()), fontsize = inp['mv_labelsize'], va = 'center', ha = 'right', fontweight = 'bold')
        
        # Set full label
        if inp['mv_label_vip'] == True:
            labels2 = [inp['mapper_name'][label] for label in df_hold['Component Name']]
        else:
            labels2 = [label for label in df_hold['Component Name']]

        ax2.set_yticklabels(labels2, fontsize = inp['mv_labelsize'], va = 'center', ha = 'left', fontweight = 'bold')
        # Set label
        ax1.set_xlabel(f'VIP score', fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_ylabel(None)

        # Legend
        legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
        # Sort both labels and handles by labels
        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        leg = ax1.legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc="lower right", frameon = False, prop = legendproperties)
        
        # Name append
        if inp['mv_scaling'] == True:
            app_scale = f'_{inp["mv_scaling_method"]}'
        else:
            app_scale = ''

        list_bbox = [leg]
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'vip_pls_SN{inp["pre_signal_noise"]}_relevant{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'vip_pls_SN{inp["pre_signal_noise"]}_relevant{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
        plt.close(fig)
    return

def scan_mv_pls_beta_plot_full(df_beta, labels_group, inp):
    """
    Plot beta coefficients.

    Plotting function for beta coefficients scores.

    Parameters
    ----------
    df_beta : dataframe
        Dataframe with beta coefficients to plot.
    labels_group : list
        Group labels.
    inp : dict
        Method dictionary.
    """
    if len(labels_group) > 2:
        for item in labels_group:
            df_hold = df_beta.filter(['Component Name', item+'_mean', item+'_ci_lower', item+'_ci_upper', item+'_interval']).copy()
            
            # Sort by biggest beta coefficients, positive and negative
            df_sort = df_hold.copy()
            df_sort[item+'_mean'] = abs(df_sort[item+'_mean'])
            df_sort = df_sort.sort_values(by = [item+'_mean'], ascending = False).copy().reset_index(drop=True)
            df_sort['sorter'] = range(len(df_sort))
            mapper_sorting = dict(zip(df_sort['Component Name'], df_sort['sorter']))

            df_hold['sorter'] = df_hold['Component Name'].map(mapper_sorting)
            df_hold = df_hold.copy().sort_values(by = ['sorter'], ascending = False).reset_index(drop=True)
            df_hold = df_hold.copy().filter([item for item in df_hold.columns if item != 'sorter'])

            x1 = len(df_hold)
            df_hold = df_hold.loc[len(df_hold)-inp['mv_vip_number']:].copy()
            x2 = len(df_hold)

            # Create plot   
            fig = plt.figure(figsize = inp['mv_figsize_vip'])
            # Add plots for multiple axes
            ax1=fig.add_subplot(111, label="1", frameon = True)
            ax2=fig.add_subplot(111, label="2", frameon = False)
            ax2.get_xaxis().set_visible(False)
            ax2.yaxis.tick_right()
            #ax2.get_yaxis().set_visible(False)     
            
            # Plot data
            xerr = [df_hold[item+'_mean']-df_hold[item+'_ci_lower'], df_hold[item+'_ci_upper']-df_hold[item+'_mean']]
            color = inp['mapper_group_color'][item]
            markersize = 4
            points = ax1.errorbar(x = df_hold[item+'_mean'], y = df_hold.index, xerr=xerr, linewidth = 0.8, fmt = 'o', markersize = markersize, elinewidth=1, capsize=markersize, label = item)
            if 0 in df_hold[item+'_interval']:
                ax1.axvline(0, linewidth = 0.8, color = 'red')

            # Set grid
            ax1.set_axisbelow(True)
            ax1.yaxis.grid()        

            # Scale axis
            ax1.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)
            ax2.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)

            plt.locator_params(axis='x', nbins=5)
            ax1.yaxis.set_ticks(df_hold.index)
            ax2.yaxis.set_ticks(df_hold.index)
            ax1.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
            ax2.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
            ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
            ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])

            plt.draw()

            # Set standard label
            labels1 = [label for label in df_hold['Component Name']]
            ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = inp['mv_labelsize'], fontweight = 'bold')
            ax1.set_yticklabels(labels1, fontsize = inp['mv_labelsize'], va = 'center', ha = 'right', fontweight = 'bold')
            
            # Set full label
            if inp['mv_label_vip'] == True:
                labels2 = [inp['mapper_name'][label] for label in df_hold['Component Name']]
                ax2.set_yticklabels(labels2, fontsize = inp['mv_labelsize'], va = 'center', ha = 'left', fontweight = 'bold')

            # Set label
            ax1.set_xlabel(r'Beta coefficients, $\beta$', fontsize = inp['mv_labelsize'], fontweight = 'bold')
            ax1.set_ylabel(None)

            # Add non displayed data information
            legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
            list_bbox = []

            handles_class = mlines.Line2D([], [], marker=None, markersize=0, label=f'{item}')   
            leg1 = ax1.legend(handles = [handles_class], handlelength = False, bbox_to_anchor=(0, 1), loc="lower left", frameon = False, prop = legendproperties)
            list_bbox.append(leg1)

            if x1-x2 > 0:
                handles_annotation = mlines.Line2D([], [], marker=None,markersize=0, label=f'Beta Top {inp["mv_vip_number"]}')
                leg2 = ax2.legend(handles = [handles_annotation], handlelength = False, bbox_to_anchor=(1, 1), loc="lower right", frameon = False, prop = legendproperties)
                list_bbox.append(leg2)

            # Name append       
            if inp['mv_scaling'] == True:
                app_scale = f'_{inp["mv_scaling_method"]}'
            else:
                app_scale = ''
            fig.savefig(inp['path_evaluation_mv'].joinpath(f'beta_pls_{item}_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
            fig.savefig(inp['path_evaluation_mv'].joinpath(f'beta_pls_{item}_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
            plt.close(fig)
    else:
        df_hold = df_beta.filter(['Component Name', 'mean', 'ci_lower', 'ci_upper', 'interval']).copy()
            
        # Sort by biggest beta coefficients, positive and negative
        df_sort = df_hold.copy()
        df_sort['mean'] = abs(df_sort['mean'])
        df_sort = df_sort.sort_values(by = ['mean'], ascending = False).copy().reset_index(drop=True)
        df_sort['sorter'] = range(len(df_sort))
        mapper_sorting = dict(zip(df_sort['Component Name'], df_sort['sorter']))

        df_hold['sorter'] = df_hold['Component Name'].map(mapper_sorting)
        df_hold = df_hold.copy().sort_values(by = ['sorter'], ascending = False).reset_index(drop=True)
        df_hold = df_hold.copy().filter([item for item in df_hold.columns if item != 'sorter'])

        x1 = len(df_hold)
        df_hold = df_hold.loc[len(df_hold)-inp['mv_vip_number']:].copy()
        x2 = len(df_hold)

        # Create plot   
        fig = plt.figure(figsize = inp['mv_figsize_vip'])
        # Add plots for multiple axes
        ax1=fig.add_subplot(111, label="1", frameon = True)
        ax2=fig.add_subplot(111, label="2", frameon = False)
        ax2.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        #ax2.get_yaxis().set_visible(False)     
        
        # Plot data
        xerr = [df_hold['mean']-df_hold['ci_lower'], df_hold['ci_upper']-df_hold['mean']]
        markersize = 4
        points = ax1.errorbar(x = df_hold['mean'], y = df_hold.index, xerr=xerr, linewidth = 0.8, fmt = 'o', markersize = markersize, elinewidth=1, capsize=markersize)
        if 0 in df_hold['interval']:
            ax1.axvline(0, linewidth = 0.8, color = 'red')

        # Set grid
        ax1.set_axisbelow(True)
        ax1.yaxis.grid()        

        # Scale axis
        ax1.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)
        ax2.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)

        plt.locator_params(axis='x', nbins=5)
        ax1.yaxis.set_ticks(df_hold.index)
        ax2.yaxis.set_ticks(df_hold.index)
        ax1.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])

        plt.draw()

        # Set standard label
        labels1 = [label for label in df_hold['Component Name']]
        ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_yticklabels(labels1, fontsize = inp['mv_labelsize'], va = 'center', ha = 'right', fontweight = 'bold')
        
        # Set full label
        if inp['mv_label_vip'] == True:
            labels2 = [inp['mapper_name'][label] for label in df_hold['Component Name']]
            ax2.set_yticklabels(labels2, fontsize = inp['mv_labelsize'], va = 'center', ha = 'left', fontweight = 'bold')

        # Set label
        ax1.set_xlabel(r'Beta coefficients, $\beta$', fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_ylabel(None)

        # Add non displayed data information
        legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
        list_bbox = []

        if x1-x2 > 0:
            handles_annotation = mlines.Line2D([], [], marker=None,markersize=0, label=f'Beta Top {inp["mv_vip_number"]}')
            leg2 = ax2.legend(handles = [handles_annotation], handlelength = False, bbox_to_anchor=(1, 1), loc="lower right", frameon = False, prop = legendproperties)
            list_bbox.append(leg2)

        # Name append       
        if inp['mv_scaling'] == True:
            app_scale = f'_{inp["mv_scaling_method"]}'
        else:
            app_scale = ''
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'beta_pls_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'beta_pls_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
        plt.close(fig)
    return

def scan_mv_pls_beta_plot_relevant(df_beta, labels_group, inp):
    """
    Plot beta coefficients.

    Plotting function for user-specified beta coefficients scores.

    Parameters
    ----------
    df_beta : dataframe
        Dataframe with beta coefficients to plot.
    labels_group : list
        Group labels.
    inp : dict
        Method dictionary.
    """
    if len(labels_group) > 2:
        # Create plot   
        fig = plt.figure(figsize = inp['mv_figsize_vip'])
        # Add plots for multiple axes
        ax1=fig.add_subplot(111, label="1", frameon = True)
        ax2=fig.add_subplot(111, label="2", frameon = False)
        ax2.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        #ax2.get_yaxis().set_visible(False)   

        mapper_sorting = dict(zip(inp['pre_list_relevant'],range(len(inp['pre_list_relevant']))))
        df_beta = df_beta[df_beta['Component Name'].isin(set(inp['pre_list_relevant']))].copy().reset_index(drop=True)
        df_beta['sorter'] = df_beta['Component Name'].map(mapper_sorting)
        df_beta = df_beta.sort_values(by = ['sorter'], ascending = False).copy().reset_index(drop=True)
        df_beta = df_beta.filter([item for item in df_beta.columns if item != 'sorter']).copy()

        for item in labels_group:
            df_hold = df_beta.filter(['Component Name', item+'_mean', item+'_ci_lower', item+'_ci_upper', item+'_interval', item+'_relevant']).copy()

            color = inp['mapper_group_color'][item]
            markersize = 3

            df_rel = df_hold[df_hold[item+'_relevant'] == True].copy()
            xerr_rel = [df_rel[item+'_mean']-df_rel[item+'_ci_lower'], df_rel[item+'_ci_upper']-df_rel[item+'_mean']]
            ax1.errorbar(x = df_rel[item+'_mean'], y = df_rel.index, xerr=xerr_rel, linewidth = 0.8, color = color, fmt = 'o', ecolor=color, markersize = markersize, elinewidth=1, capsize=markersize, label = item+r' ($\beta$ $\neq$ 0)')
            
            df_notrel = df_hold[df_hold[item+'_relevant'] == False].copy()
            xerr_notrel = [df_notrel[item+'_mean']-df_notrel[item+'_ci_lower'], df_notrel[item+'_ci_upper']-df_notrel[item+'_mean']]
            ax1.errorbar(x = df_notrel[item+'_mean'], y = df_notrel.index, xerr=xerr_notrel, linewidth = 0.8, color = 'grey', fmt = 'o', ecolor='grey', markersize = markersize, elinewidth=1, capsize=markersize, label = item+r' ($\beta$ = 0)')

            if 0 in df_hold[item+'_interval']:
                ax1.axvline(0, linewidth = 0.8, color = 'red')

        # Set grid
        ax1.set_axisbelow(True)
        ax1.yaxis.grid()        
        plt.locator_params(axis = 'x', nbins=4)

        # Scale axis
        ax1.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)
        ax2.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)

        plt.locator_params(axis='x', nbins=5)
        ax1.yaxis.set_ticks(df_hold.index)
        ax2.yaxis.set_ticks(df_hold.index)
        ax1.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        
        plt.draw()

        # Set standard label
        #labels1 = [label for label in df_hold['Component Name']]
        ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_yticklabels([None]*len(ax1.get_yticklabels()), fontsize = inp['mv_labelsize'], va = 'center', ha = 'right', fontweight = 'bold')
        
        # Set full label
        if inp['mv_label_vip'] == True:
            labels2 = [inp['mapper_name'][label] for label in df_hold['Component Name']]
        else:
            labels2 = [label for label in df_hold['Component Name']]

        ax2.set_yticklabels(labels2, fontsize = inp['mv_labelsize'], va = 'center', ha = 'left', fontweight = 'bold')
        # Set label
        ax1.set_xlabel(r'Beta coefficients, $\beta$', fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_ylabel(None)

        # Legend
        legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
        # Sort both labels and handles by labels
        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        leg = ax1.legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc="lower right", frameon = False, prop = legendproperties)
        
        # Name append    
        if inp['mv_scaling'] == True:
            app_scale = f'_{inp["mv_scaling_method"]}'
        else:
            app_scale = ''

        list_bbox = [leg]
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'beta_pls_SN{inp["pre_signal_noise"]}_relevant{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'beta_pls_SN{inp["pre_signal_noise"]}_relevant{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
        plt.close(fig)
    else:
        # Create plot   
        fig = plt.figure(figsize = inp['mv_figsize_vip'])
        # Add plots for multiple axes
        ax1=fig.add_subplot(111, label="1", frameon = True)
        ax2=fig.add_subplot(111, label="2", frameon = False)
        ax2.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        #ax2.get_yaxis().set_visible(False)   

        mapper_sorting = dict(zip(inp['pre_list_relevant'],range(len(inp['pre_list_relevant']))))
        df_beta = df_beta[df_beta['Component Name'].isin(set(inp['pre_list_relevant']))].copy().reset_index(drop=True)
        df_beta['sorter'] = df_beta['Component Name'].map(mapper_sorting)
        df_beta = df_beta.sort_values(by = ['sorter'], ascending = False).copy().reset_index(drop=True)
        df_beta = df_beta.filter([item for item in df_beta.columns if item != 'sorter']).copy()

        df_hold = df_beta.filter(['Component Name', 'mean', 'ci_lower', 'ci_upper', 'interval', 'relevant']).copy()

        color = 'black'
        markersize = 3

        df_rel = df_hold[df_hold['relevant'] == True].copy()
        xerr_rel = [df_rel['mean']-df_rel['ci_lower'], df_rel['ci_upper']-df_rel['mean']]
        ax1.errorbar(x = df_rel['mean'], y = df_rel.index, xerr=xerr_rel, linewidth = 0.8, color = color, fmt = 'o', ecolor=color, markersize = markersize, elinewidth=1, capsize=markersize, label = '$\beta$ $\neq$ 0')
        
        df_notrel = df_hold[df_hold['relevant'] == False].copy()
        xerr_notrel = [df_notrel['mean']-df_notrel['ci_lower'], df_notrel['ci_upper']-df_notrel['mean']]
        ax1.errorbar(x = df_notrel['mean'], y = df_notrel.index, xerr=xerr_notrel, linewidth = 0.8, color = 'grey', fmt = 'o', ecolor='grey', markersize = markersize, elinewidth=1, capsize=markersize, label = '$\beta$ = 0')

        if 0 in df_hold['interval']:
            ax1.axvline(0, linewidth = 0.8, color = 'red')

        # Set grid
        ax1.set_axisbelow(True)
        ax1.yaxis.grid()        
        plt.locator_params(axis = 'x', nbins=4)

        # Scale axis
        ax1.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)
        ax2.set_ylim(df_hold.index[0]-1,df_hold.index[-1]+1)

        plt.locator_params(axis='x', nbins=5)
        ax1.yaxis.set_ticks(df_hold.index)
        ax2.yaxis.set_ticks(df_hold.index)
        ax1.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'both', labelsize = inp['mv_labelsize'])
        ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        
        plt.draw()

        # Set standard label
        #labels1 = [label for label in df_hold['Component Name']]
        ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_yticklabels([None]*len(ax1.get_yticklabels()), fontsize = inp['mv_labelsize'], va = 'center', ha = 'right', fontweight = 'bold')
        
        # Set full label
        if inp['mv_label_vip'] == True:
            labels2 = [inp['mapper_name'][label] for label in df_hold['Component Name']]
        else:
            labels2 = [label for label in df_hold['Component Name']]

        ax2.set_yticklabels(labels2, fontsize = inp['mv_labelsize'], va = 'center', ha = 'left', fontweight = 'bold')
        # Set label
        ax1.set_xlabel(r'Beta coefficients, $\beta$', fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax1.set_ylabel(None)

        # Legend
        legendproperties = {'size': inp['mv_labelsize'], 'weight': 'bold'}
        # Sort both labels and handles by labels
        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        leg = ax1.legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc="lower right", frameon = False, prop = legendproperties)
        
        # Name append    
        if inp['mv_scaling'] == True:
            app_scale = f'_{inp["mv_scaling_method"]}'
        else:
            app_scale = ''

        list_bbox = [leg]
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'beta_pls_SN{inp["pre_signal_noise"]}_relevant{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'beta_pls_SN{inp["pre_signal_noise"]}_relevant{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
        plt.close(fig)
    return

def scan_mv_pls_roc_plot_full(dict_roc, labels_group, inp):
    """
    Plot ROC curves.

    Plotting function for user-specified beta coefficients scores.

    Parameters
    ----------
    dict_roc : dict
        Dictionary with ROC and AUC.
    labels_group : list
        Group labels.
    inp : dict
        Method dictionary.
    """
    palette = seaborn.color_palette('colorblind', n_colors = 3)

    df_train_roc = dict_roc['TrainROC'].copy()
    df_train_auc = dict_roc['TrainAUC'].copy()
    df_test_roc = dict_roc['TestROC'].copy()
    df_test_auc = dict_roc['TestAUC'].copy()

    # Account for two classes
    if len(labels_group) == 2:
        labels_group = ["0"]

    for item in labels_group:
        auc_train_mean = numpy.round(df_train_auc.at[0, item+'_auc_mean'],5)
        auc_test_mean = numpy.round(df_test_auc.at[0, item+'_auc_mean'],5)

        auc_train_ci = numpy.round(df_train_auc.at[0, item+'_auc_std'],5)
        auc_test_ci = numpy.round(df_test_auc.at[0, item+'_auc_std'],5)

        # Create plot   
        fig, ax = plt.subplots(figsize = (8,6)) 
        ax.plot([0, 1], [0, 1], lw = 1.5, linestyle = '--', color = 'k', alpha=.8, label = 'Random classifier')
        
        # Train
        ax.plot(df_train_roc[item+'_fpr_mean'], df_train_roc[item+'_tpr_mean'], color = palette[0], lw = 1.5, linestyle = '-', label = fr'In-bag prediction (AUC = {auc_train_mean} $\pm$ {auc_train_ci})')
        ax.fill_between(df_train_roc[item+'_fpr_mean'], df_train_roc[item+'_tpr_ci_lower'], df_train_roc[item+'_tpr_ci_upper'], color = palette[0], alpha=.2, label=fr'In-bag prediction 95% CI')
        
        # Test
        ax.plot(df_test_roc[item+'_fpr_mean'], df_test_roc[item+'_tpr_mean'], color = palette[1], lw = 1.5, linestyle = '-', label = fr'Out-of-bag prediction (AUC = {auc_test_mean} $\pm$ {auc_test_ci})')
        ax.fill_between(df_test_roc[item+'_fpr_mean'], df_test_roc[item+'_tpr_ci_lower'], df_test_roc[item+'_tpr_ci_upper'], color = palette[1], alpha=.2, label=fr'Out-of-bag prediction 95% CI')
        
        ax.set_xlim(-0.05,1.05)
        ax.set_ylim(-0.05,1.05)
        ax.set_xlabel('False positive rate', fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax.set_ylabel('True positive rate', fontsize = inp['mv_labelsize'], fontweight = 'bold')
        ax.tick_params(labelsize = inp['mv_labelsize'])
        legendproperties = {'size': inp['mv_labelsize']-2, 'weight': 'bold'}
        leg1 = ax.legend(loc="upper left", bbox_to_anchor = (1,1), frameon = False, prop = legendproperties)

        # Name append    
        if inp['mv_scaling'] == True:
            app_scale = f'_{inp["mv_scaling_method"]}'
        else:
            app_scale = ''
        list_bbox = [leg1]
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'permutation_ROC_{item}_SN{inp["pre_signal_noise"]}{app_scale}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_mv'].joinpath(f'permutation_ROC_{item}_SN{inp["pre_signal_noise"]}{app_scale}.svg'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', format = 'svg')
        plt.close(fig)

    return

def scan_cluster(inp, cluster_threshold_kruskal = True, cluster_threshold_beta = True, cluster_threshold_vip = True, cluster_threshold_vip_relevant = True,  cluster_orientation = 'vertical', cluster_vip_top_number = 50, cluster_mean_area = True,  cluster_labelsize_cluster = 12, cluster_figsize_cluster = (10,20)):
    """
    Cluster analysis.

    Basis function for cluster analysis of scan data.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    cluster_threshold_kruskal : bool
        Apply Kruskal-Wallis filter.
    cluster_threshold_beta : bool
        Apply beta coefficients filter.
    cluster_threshold_vip : bool
        Apply VIP filter.
    cluster_threshold_vip_relevant : bool
        Apply VIP filter for user-specific analytes.
    cluster_vip_top_number : int
        Number of relevant VIP analytes.
    cluster_mean_area : bool
        Average areas.
    cluster_labelsize_cluster : int
        Labelsize for all plots.
    cluster_figsize_cluster : tuple
        Figsize for hierachical dendogram plot.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    if len(set(inp['data_processed_filter']['Group'])) > 1:
        # Parse parameter
        inp = scan_cluster_parsing(inp, cluster_threshold_kruskal, cluster_threshold_beta, cluster_threshold_vip, cluster_threshold_vip_relevant, cluster_orientation, cluster_vip_top_number, cluster_mean_area, cluster_labelsize_cluster, cluster_figsize_cluster)
        # Create folder
        inp = scan_cluster_folder(inp)
        # Cluster analysis
        inp = scan_cluster_main(inp)
    return inp

def scan_cluster_parsing(inp, cluster_threshold_kruskal, cluster_threshold_beta, cluster_threshold_vip, cluster_threshold_vip_relevant, cluster_orientation, cluster_vip_top_number, cluster_mean_area, cluster_labelsize_cluster, cluster_figsize_cluster):
    """
    Initialize cluster analysis.

    Initialization function for cluster analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    cluster_threshold_kruskal : bool
        Apply Kruskal-Wallis filter.
    cluster_threshold_beta : bool
        Apply beta coefficients filter.
    cluster_threshold_vip : bool
        Apply VIP filter.
    cluster_threshold_vip_relevant : bool
        Apply VIP filter for user-specific analytes.
    cluster_vip_top_number : int
        Number of relevant VIP analytes.
    cluster_mean_area : bool
        Average areas.
    cluster_labelsize_cluster : int
        Labelsize for all plots.
    cluster_figsize_cluster : tuple
        Figsize for hierachical dendogram plot.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Plots
    inp['cluster_threshold_kruskal'] = cluster_threshold_kruskal
    inp['cluster_threshold_beta'] = cluster_threshold_beta
    inp['cluster_threshold_vip'] = cluster_threshold_vip
    inp['cluster_threshold_vip_relevant'] = cluster_threshold_vip_relevant
    inp['cluster_orientation'] = cluster_orientation
    inp['cluster_vip_top_number'] = cluster_vip_top_number
    inp['cluster_mean_area'] = cluster_mean_area
    inp['cluster_labelsize'] = cluster_labelsize_cluster
    inp['cluster_figsize_hierarchical'] = cluster_figsize_cluster
    return inp

def scan_cluster_folder(inp):
    """
    Create folder.

    Create folder for cluster analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    inp['path_evaluation_cluster'] = create_folder(inp['path_evaluation'], '04_cluster')
    return inp

def scan_cluster_main(inp):
    """
    Calculate cluster analysis.

    Main function for cluster analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Allocate data
    df = inp['data_processed_filter'].copy()
    df = df[~df['Class'].isin(['QC'])].copy()

    # Cluster heatmap Features/Samples (full)
    scan_cluster_FS_full(df, inp)

    # Cluster heatmap of predefined list
    if (len(inp['pre_list_relevant'])!=0) & (inp['cluster_threshold_vip_relevant'] == True):
        scan_cluster_FS_single_relevant(df, inp)
    
    return inp

def scan_cluster_FS_full(df, inp):
    """
    Full hierarchical clustering.

    Get hierarchical clustering for all analytes with user-specific filters.

    Parameters
    ----------
    df : dataframe
        Preprocessed data.
    inp : dict
        Method dictionary.
    """
    # Clustermap parameter
    cmapGR = LinearSegmentedColormap.from_list(name = 'clustermap', colors=['black','green'])
    # Cycle samples
    df_clustermap = df.copy()
    if inp['cluster_threshold_kruskal'] == True:
        df_statistics_test_kruskal = inp['uv_statistic_kruskal'].copy()
        df_statistics_test_kruskal = df_statistics_test_kruskal[df_statistics_test_kruskal['significant center'] == True].copy()
        set_kruskal = set(df_statistics_test_kruskal['Component Name'])
        append_kruskal = '_kruskal'
    else:
        set_kruskal = set()
        append_kruskal = ''

    if inp['cluster_threshold_beta'] == True:
        # Select group
        df_beta = inp['predictor_relevant_beta'].copy() 
        set_beta = set(df_beta['Component Name'])
        append_beta = '_beta'
    else:
        set_beta = set()
        append_beta = ''

    if inp['cluster_threshold_vip'] == True:
        # Select group
        df_vip = inp['predictor_relevant_vip'].copy() 
        set_vip = set(df_vip['Component Name'])
        append_vip = '_vip'
    else:
        set_vip = set()
        append_vip = ''
    
    set_all = set_kruskal&set_beta&set_vip

    if not set_all == set():
        df_clustermap = df_clustermap[df_clustermap['Component Name'].isin(set_all)].copy()

    df_filter = df_clustermap.copy()
    df_pivot = scan_cluster_preprocessing(df_filter, inp).copy().T
    
    # Save plots
    if inp['mv_scaling'] == True:
        append_scale = f'_{inp["mv_scaling_method"]}'
    else:
        append_scale = ''

    if inp['cluster_mean_area'] == True:
        append_mean = f'_mean'
    else:
        append_mean = ''
    
    save_df(df_pivot, inp['path_evaluation_cluster'], f'data_pivot{append_kruskal}{append_vip}{append_beta}{append_scale}{append_mean}',index=True)
    # Get dendogram species and set group color
    used_groups = [item for item in df_pivot.columns.get_level_values('Group').unique() if item.lower() != 'qc']
    used_columns = (df_pivot.columns.get_level_values('Group').astype(str).isin(used_groups))
    # Identify groups
    group_pal = inp['mapper_group_color']
    group_lut = group_pal
    group_labels = df_pivot.columns.get_level_values('Group')
    group_colors = pandas.Series(group_labels, index=df_pivot.columns).map(group_lut)


    if inp['cluster_orientation'].lower() == 'vertical':
        # Plot clustermap
        cm = seaborn.clustermap(
            data = df_pivot, method = 'average', metric = 'correlation', 
            standard_scale=0, col_colors = group_colors, cmap = cmapGR, 
            linewidths=0, xticklabels = 1, yticklabels = 1, figsize = inp['cluster_figsize_hierarchical'], tree_kws=dict(linewidths=1.5))
        
        # Format plot
        labels = [inp['mapper_name'][label.get_text()] for label in cm.ax_heatmap.get_yticklabels()]
        cm.ax_heatmap.set_xticks([])
        cm.ax_heatmap.set_xticklabels(len(cm.ax_heatmap.get_xticklabels())*[None], fontsize = inp['cluster_labelsize'], fontweight = 'bold')
        cm.ax_heatmap.set_yticklabels(labels, fontsize = inp['cluster_labelsize'], fontweight = 'bold', rotation = 0)
        cm.ax_col_colors.set_yticklabels(['Group'], fontsize = inp['cluster_labelsize']+2, fontweight = 'bold')
        cm.ax_heatmap.set_xlabel(None, fontsize = inp['cluster_labelsize'], fontweight = 'bold')
        cm.ax_heatmap.set_ylabel(None, fontsize = inp['cluster_labelsize'], fontweight = 'bold')
        cm.ax_cbar.set_ylabel('Relative area', fontsize = inp['cluster_labelsize'], fontweight = 'bold')

        # Create legend
        legendproperties = {'size': inp['cluster_labelsize'], 'weight': 'bold'}
        handles = [Patch(facecolor=group_pal[name]) for name in sorted(used_groups)]
        labels = sorted(used_groups)
        leg = plt.legend(
            handles, labels, ncol = 2,
            bbox_transform = cm.ax_heatmap.transAxes, bbox_to_anchor=(0,0), borderaxespad = 0,
            loc='upper left', frameon = False, prop = legendproperties)

        list_analytes = [
            mlines.Line2D([],[], linewidth = 0, label = f'{int(len(df_pivot.index))} analytes'),
            mlines.Line2D([],[], linewidth = 0, label = f'{int(len(df_pivot.columns))} samples')
            ]

        n_analytes = cm.ax_heatmap.legend(
            handles = list_analytes, 
            bbox_transform = cm.ax_heatmap.transAxes, bbox_to_anchor=(1.0, 0), borderaxespad = 0, 
            loc="upper right", frameon = False, prop = legendproperties)

    else:
        # Plot clustermap
        cm = seaborn.clustermap(
            data = df_pivot.T, method = 'average', metric = 'correlation', 
            standard_scale=1, row_colors = group_colors, cmap = cmapGR, 
            linewidths=0, xticklabels = 1, yticklabels = 1, figsize = inp['cluster_figsize_hierarchical'], tree_kws=dict(linewidths=1.5))
        
        # Format plot
        labels = [inp['mapper_name'][label.get_text()] for label in cm.ax_heatmap.get_xticklabels()]
        cm.ax_heatmap.set_yticks([])
        cm.ax_heatmap.set_xticklabels(labels, fontsize = inp['cluster_labelsize'], fontweight = 'bold', rotation = 45, ha = 'right')
        cm.ax_heatmap.set_yticklabels(len(cm.ax_heatmap.get_yticklabels())*[None], fontsize = inp['cluster_labelsize'], fontweight = 'bold')
        cm.ax_row_colors.set_xticklabels(['Group'], fontsize = inp['cluster_labelsize']+2, fontweight = 'bold', rotation = 45, ha = 'right')
        cm.ax_heatmap.set_xlabel(None, fontsize = inp['cluster_labelsize'], fontweight = 'bold')
        cm.ax_heatmap.set_ylabel(None, fontsize = inp['cluster_labelsize'], fontweight = 'bold')
        cm.ax_cbar.set_ylabel('Relative area', fontsize = inp['cluster_labelsize'], fontweight = 'bold')

        # Create legend
        legendproperties = {'size': inp['cluster_labelsize'], 'weight': 'bold'}
        handles = [Patch(facecolor=group_pal[name]) for name in sorted(used_groups)]
        labels = sorted(used_groups)
        leg = plt.legend(
            handles, labels, ncol = 2,
            bbox_transform = cm.ax_heatmap.transAxes, bbox_to_anchor=(1,1), borderaxespad = 0,
            loc='upper left', frameon = False, prop = legendproperties)

        list_analytes = [
            mlines.Line2D([],[], linewidth = 0, label = f'{int(len(df_pivot.index))} analytes'),
            mlines.Line2D([],[], linewidth = 0, label = f'{int(len(df_pivot.columns))} samples')
            ]

        n_analytes = cm.ax_heatmap.legend(
            handles = list_analytes, 
            bbox_transform = cm.ax_heatmap.transAxes, bbox_to_anchor=(1.0, 0), handlelength = 0, borderaxespad = 0, 
            loc="lower left", frameon = False, prop = legendproperties)

    # Save plots
    list_bbox = [cm.ax_cbar, cm.ax_col_dendrogram, cm.ax_heatmap, cm.ax_row_dendrogram, leg, n_analytes]
    cm.fig.savefig(inp['path_evaluation_cluster'].joinpath(f'cluster_all_SN{inp["pre_signal_noise"]}{append_kruskal}{append_vip}{append_beta}{append_scale}{append_mean}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
    cm.fig.savefig(inp['path_evaluation_cluster'].joinpath(f'cluster_all_SN{inp["pre_signal_noise"]}{append_kruskal}{append_vip}{append_beta}{append_scale}{append_mean}.svg'), format='svg' , bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
    plt.close(cm.fig)
    return

def scan_cluster_FS_single_relevant(df, inp):
    """
    Relevant hierarchical clustering.

    Get hierarchical clustering for relevant analytes with user-specific filters.

    Parameters
    ----------
    df : dataframe
        Preprocessed data.
    inp : dict
        Method dictionary.
    """
    # Allocate data
    # Clustermap parameter
    cmapGR = LinearSegmentedColormap.from_list(name = 'clustermap', colors=['black','green'])
    # Cycle samples
    df_clustermap = df.copy()
    # Select group
    df_filter = df_clustermap[df_clustermap['Component Name'].isin(set(inp['pre_list_relevant']))].copy()
    df_pivot = scan_cluster_preprocessing(df_filter).copy().T
    # Get dendogram species and set group color
    used_groups = [item for item in df_pivot.columns.get_level_values('Group').unique() if item.lower() != 'qc']
    used_columns = (df_pivot.columns.get_level_values('Group').astype(str).isin(used_groups))
    # Identify groups
    group_pal = inp['mapper_group_color']
    group_lut = group_pal
    group_labels = df_pivot.columns.get_level_values('Group')
    group_colors = pandas.Series(group_labels, index=df_pivot.columns).map(group_lut)
    # Plot clustermap
    cm = seaborn.clustermap(
        data = df_pivot, method = 'average', metric = 'correlation', 
        standard_scale=0, col_colors = group_colors, cmap = cmapGR, 
        linewidths=0, xticklabels = 1, yticklabels = 1, figsize = inp['cluster_figsize_hierarchical'], tree_kws=dict(linewidths=1.5))
    # Format plot
    labels = [inp['mapper_name'][label.get_text()] for label in cm.ax_heatmap.get_yticklabels()]
    cm.ax_heatmap.set_xticks([])
    cm.ax_heatmap.set_xticklabels(len(cm.ax_heatmap.get_xticklabels())*[None], fontsize = inp['cluster_labelsize'], fontweight = 'bold')
    cm.ax_heatmap.set_yticklabels(labels, fontsize = inp['cluster_labelsize'], fontweight = 'bold', rotation = 0)
    cm.ax_col_colors.set_yticklabels(['Group'], fontsize = inp['cluster_labelsize']+2, fontweight = 'bold')
    cm.ax_heatmap.set_xlabel(None, fontsize = inp['cluster_labelsize'], fontweight = 'bold')
    cm.ax_heatmap.set_ylabel(None, fontsize = inp['cluster_labelsize'], fontweight = 'bold')
    cm.ax_cbar.set_ylabel('Scaled value', fontsize = inp['cluster_labelsize'], fontweight = 'bold')
    # Create legend
    legendproperties = {'size': inp['cluster_labelsize'], 'weight': 'bold'}
    handles = [Patch(facecolor=group_pal[name]) for name in sorted(used_groups)]
    labels = sorted(used_groups)
    leg = plt.legend(
        handles, labels, ncol = len(labels),
        bbox_transform = cm.ax_heatmap.transAxes, bbox_to_anchor=(0, 0), borderaxespad = 0,
        loc='upper left', frameon = False, prop = legendproperties)

    list_analytes = [
        mlines.Line2D([],[], linewidth = 0, label = f'{int(len(df_pivot.index))} analytes'),
        mlines.Line2D([],[], linewidth = 0, label = f'{int(len(df_pivot.columns))} samples')
        ]

    n_analytes = cm.ax_heatmap.legend(
        handles = list_analytes, 
        bbox_transform = cm.ax_heatmap.transAxes, bbox_to_anchor=(1.0, 0), borderaxespad = 0, 
        loc="upper right", frameon = False, prop = legendproperties)
    # Save plots
    if inp['mv_scaling'] == True:
        append_scale = f'_{inp["mv_scaling_method"]}'
    else:
        append_scale = ''

    if inp['cluster_mean_area'] == True:
        append_mean = f'_mean'
    else:
        append_mean = ''

    list_bbox = [cm.ax_cbar, cm.ax_col_dendrogram, cm.ax_heatmap, cm.ax_row_dendrogram, leg, n_analytes]
    cm.fig.savefig(inp['path_evaluation_cluster'].joinpath(f'cluster_SN{inp["pre_signal_noise"]}_relevant{append_scale}{append_mean}.png'), bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
    cm.fig.savefig(inp['path_evaluation_cluster'].joinpath(f'cluster_SN{inp["pre_signal_noise"]}_relevant{append_scale}{append_mean}.svg'), format='svg' , bbox_extra_artists = (list_bbox), bbox_inches = 'tight', dpi = 800)
    plt.close(cm.fig)
    return

def scan_cluster_preprocessing(df, inp):
    """
    Cluster preprocessing.

    Preprocess data for hierarchical clustering.

    Parameters
    ----------
    df : dataframe
        Preprocessed data.
    inp : dict
        Method dictionary.

    Returns
    -------
    df_pivot : dataframe
        Pivot data.
    """
    # Format multivariate matrix
    df_filter = df.filter(['Area','Component Name','Group']).drop_duplicates()
    
    if inp['cluster_mean_area'] == True:
        df_pivot = df_filter.pivot_table(index = ['Group'], columns = 'Component Name', values = 'Area').copy()
    else:
        df_filter['Sample'] = df_filter.groupby(['Group', 'Component Name']).cumcount()
        df_pivot = df_filter.pivot_table(index = ['Group','Sample'], columns = 'Component Name', values = 'Area').copy()
    
    df_pivot = df_pivot.apply(pandas.to_numeric).copy()
    df_pivot = scan_mv_preprocessing_fill_pivot_nan(df_pivot)

    return df_pivot

def scan_pathway(inp, path_org, pathway_alpha = 0.05, pathway_correction = 'Holm-Bonferroni', pathway_selection = 'univariate', pathway_measure = 'degree',  pathway_labelsize_pathway = 12, pathway_figsize_pathway = (6,8), pathway_number_pathways_top = 20):
    """
    Pathway analysis.

    Basis function for pathway analysis of scan data.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    path_org : str
        Raw path string leading to the corresponding pathway data, generated by DSFIApy database.
    pathway_alpha : float
        Probability of error.
    pathway_correction : str
        Multi comparison correction.
    pathway_measure : str
        Topology measure.
    pathway_labelsize_pathway : int
        Labelsize for all plots.
    pathway_figsize_pathway : tuple
        Figsize for all plots.
    pathway_number_pathways_top : int
        Number of pathways to display.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    if len(set(inp['data_processed_filter']['Group'])) > 1:
        # Parse parameter
        inp = scan_pathway_parsing(inp, path_org, pathway_alpha, pathway_correction, pathway_selection, pathway_measure,  pathway_labelsize_pathway, pathway_figsize_pathway, pathway_number_pathways_top)
        # Create folder
        inp = scan_pathway_folder(inp)
        # Pathway analysis
        inp = scan_pathway_main(inp)
    return inp

def scan_pathway_parsing(inp, path_org, pathway_alpha, pathway_correction, pathway_selection, pathway_measure, pathway_labelsize_pathway, pathway_figsize_pathway, pathway_number_pathways_top):
    """
    Initialize pathway analysis.

    Initialization function for pathway analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    path_org : str
        Raw path string leading to the corresponding pathway data, generated by DSFIApy database.
    pathway_alpha : float
        Probability of error.
    pathway_correction : str
        Multi comparison correction.
    pathway_measure : str
        Topology measure.
    pathway_labelsize_pathway : int
        Labelsize for all plots.
    pathway_figsize_pathway : tuple
        Figsize for all plots.
    pathway_number_pathways_top : int
        Number of pathways to display.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Paths
    inp['path_file_pathway_org'] = Path(path_org)
    # Parameter
    inp['pathway_alpha'] = pathway_alpha
    inp['pathway_correction'] = pathway_correction
    inp['pathway_selection'] = pathway_selection
    inp['pathway_topology_measure'] = pathway_measure
    # Plots
    inp['pathway_labelsize'] = pathway_labelsize_pathway
    inp['pathway_figsize'] = pathway_figsize_pathway
    inp['pathway_number_top'] = pathway_number_pathways_top
    # Pathways
    inp['information_pathway_org'] = pandas.read_excel(inp['path_file_pathway_org'], sheet_name = None)
    return inp

def scan_pathway_folder(inp):
    """
    Create folder.

    Create folder for pathway analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    inp['path_evaluation_pathway'] = create_folder(inp['path_evaluation'], '05_pathway')
    inp['path_evaluation_ora'] = create_folder(inp['path_evaluation_pathway'], '01_ORA')
    inp['path_evaluation_msea'] = create_folder(inp['path_evaluation_pathway'], '02_MSEA')
    inp['path_evaluation_topology'] = create_folder(inp['path_evaluation_pathway'], '03_Topology')
    return inp

def scan_pathway_main(inp):
    """
    Calculate pathway analysis.

    Main function for pathway analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Prepare pathway analysis
    inp = scan_pathway_preparation(inp)
    
    # Over representation analysis, ORA
    print('Over Representation Analysis (ORA)')
    inp = scan_pathway_ora(inp)

    # Quantitative enrichment analysis, QEA
    print('Quantitative Enrichment Analysis (QEA)')
    inp = scan_pathway_msea(inp)

    # Pathway topology analysis
    print('Pathway Topology Analysis')
    inp = scan_pathway_topology(inp)
    
    # Plotting ORA
    print('Plotting ORA')
    scan_pathway_ora_plot(inp)

    # Plotting QEA
    print('Plotting QEA')
    scan_pathway_msea_plot(inp)

    # Plotting Topology Analysis
    print('Plotting Topology Analysis')
    scan_pathway_topology_plot(inp)
    return inp

def scan_pathway_preparation(inp):
    """
    Prepare pathway analysis.

    Create basic network files for further analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    # Allocate data
    df_nodes_raw = inp['information_pathway_org']['compounds'].copy()
    df_edges_raw = inp['information_pathway_org']['network_reactions'].copy()

    # Get pathways for pathway analysis
    pathways_all = df_edges_raw['pathway_id'].unique()

    # Remove summary pathways
    pathways_select = [item for item in pathways_all if int(item[-5:]) < 1100]
    
    # Preallocation
    list_edges = []

    # Cycle organisms
    for org_id in set(df_edges_raw['organism_id']):
        df_org = df_edges_raw[df_edges_raw['organism_id'] == org_id].copy()
        org_name = df_org['organism_name'].unique()[0]
        # Cycle pathways
        for pathway_id in pathways_select:
            df_pathway = df_org[df_org['pathway_id'] == pathway_id].copy()
            pathway_name = df_pathway['pathway_name'].unique()[0]
            # Cycle reactions
            for reaction_id in set(df_pathway['reaction_id']):
                # Get substrates and products
                df_reaction = df_pathway[df_pathway['reaction_id'] == reaction_id].copy()
                substrates = set(df_reaction['substrate'])
                products = set(df_reaction['product'])
                # Get reaction type
                reaction_type = df_reaction['type'].unique()[0]
                edges = get_product_combinations_two_sets(substrates, products)
                # Cycle edges
                for edge in edges:
                    list_edges.append([edge[0], edge[1], org_id, org_name, pathway_id, pathway_name, reaction_id, reaction_type])

    # Create dataframe
    df_edges = pandas.DataFrame(list_edges, columns = ['substrate', 'product', 'organism_id', 'organism_name', 'pathway_id', 'pathway_name', 'reaction_id', 'type']).sort_values(by=['substrate','product'])
    df_edges = df_edges.fillna('None')
    
    # Remove irreversible reactions [A -> B] if reversible reaction [A <-> B] exists
    df_edges = df_edges.sort_values(['type'], ascending = False)
    df_edges = df_edges.drop_duplicates(['substrate','product','pathway_id'], keep = 'first')
    df_nodes = df_nodes_raw.copy().sort_values(by=['compound_id'])
    df_nodes = df_nodes.fillna('None')
    
    # Get intersect
    df_nodes = df_nodes.filter(['id','compound_id','compound_name']).copy()
    df_nodes = df_nodes.drop_duplicates(subset=['compound_id']).copy()
    df_edges = df_edges[(df_edges['substrate'].isin(set(df_nodes['compound_id'])))&(df_edges['product'].isin(set(df_nodes['compound_id'])))].copy()

    # Creat mapper
    inp['mapper_pathway'] = dict(zip(df_edges['pathway_id'], df_edges['pathway_name']))

    # Create dictionary
    dict_network = {}
    dict_network['nodes'] = df_nodes.copy()
    dict_network['edges'] = df_edges.copy()

    inp['pathway_analysis_data'] = dict_network.copy()
    save_dict(dict_network, inp['path_evaluation_pathway'], 'data_pathway_raw', single_files=False, index=False)
    return inp

def scan_pathway_ora(inp):
    """
    Over representation analysis.

    Calculate fisher's exact test for over representation analysis.
    Metabolite list (e.g. expression change > 2 fold)
    Are any metabolite sets surprisingly enriched (or depleted) in my metabolite list?
    Statistical test: Fisher's Exact Test (aka Hypergeometric test)

    N = total number of metabolites in the metabolom
    M = total number of significant metabolites

    k = number of metabolites in the pathway
    n = number of significant metabolites in the pathway

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """
    
    # Allocate data
    df_filter = inp['uv_statistic_posthoc'].copy()
    
    # Get network data
    df_nodes = inp['pathway_analysis_data']['nodes'].copy()
    df_edges = inp['pathway_analysis_data']['edges'].copy()

    # Get all metabolites of organism
    metabolites_organism = set(df_edges['substrate'])|set(df_edges['product'])

    # Get all pathways of organism
    pathways_organism = set(df_edges['pathway_id'])

    # Get comparisons
    comparisons = inp['comparisons']

    # Cycle combinations
    list_ora = []
    for comparison in comparisons:
        # Univariate selection list
        if inp['pathway_selection'] == 'univariate':
            df_comp = df_filter[df_filter['group1'].isin(list(comparison)) & df_filter['group2'].isin(list(comparison))].copy()
            df_comp = df_comp[df_comp['significant center']==True].copy()
            metabolites_significant = set(df_comp['Component Name'])
            append_selection = '_uni'
        # Multivariate selection list
        else:
            df_beta = inp['predictor_relevant_beta'].copy() 
            df_vip = inp['predictor_relevant_vip'].copy()
            if len(comparisons) == 1:
                df_beta_selection = df_beta.copy()
                df_vip_selection = df_vip.copy()
            else:
                df_beta_selection = df_beta[df_beta['group'].isin(list(comparison))].copy()
                df_vip_selection = df_vip[df_vip['group'].isin(list(comparison))].copy()
            metabolites_significant = set(df_beta_selection['Component Name'])|set(df_vip_selection['Component Name'])
            append_selection = '_multi'
        # Cycle pathway
        for pathway_id in pathways_organism:
            df_pathway = df_edges[df_edges['pathway_id'] == pathway_id].copy()
            metabolites_pathway = set(df_pathway['substrate'])|set(df_pathway['product'])
            metabolites_pathway_significant = metabolites_significant & metabolites_pathway

            # Allocate variables
            N = len(metabolites_organism) ### N
            K = len(metabolites_significant) ### k == g

            n = len(metabolites_pathway) ### m
            k = len(metabolites_pathway_significant) ### x

            if (n >= 2)&(k >= 1):
                # Hit rate, Percent of the metabolite selection involved in the pathway.
                hr = k / K

                # Background hit rate, Random expectation
                bhr = n / N
                expected_x = K * bhr

                # Enrichment factor
                ef = k / expected_x

                # Calculate p-value
                p_pw = 1 - scipy.stats.hypergeom.cdf(k-1, N, n, K)

                # Append test
                list_ora.append(
                    [
                        comparison[0], comparison[1], 
                        pathway_id, inp['mapper_pathway'][pathway_id], 
                        N, K, 
                        n, k, 
                        bhr, hr, expected_x, ef, 
                        p_pw
                    ]
                    )
        
    # Create dataframe
    df_ora_hold = pandas.DataFrame(
        list_ora, 
        columns = [
            'group1', 'group2', 
            'pathway_id', 'pathway_name', 
            'metabolites_organism', 'metabolites_significant', 
            'metabolites_pathway', 'metabolites_pathway_significant',
            'background_hit_rate', 'hit_rate', 'expected', 'enrichment_factor',
            'p-value',
            ]
        )
        
    # Multi-comparison correction
    df_ora = pandas.DataFrame()
    # Cycle comparisons
    for comparison in comparisons:
        df_comp = df_ora_hold[df_ora_hold['group1'].isin(list(comparison)) & df_ora_hold['group2'].isin(list(comparison))].copy()
        df_comp['adjusted p-value'] = multiple_testing_correction(pvalues = list(df_comp['p-value']), correction_type=inp['pathway_correction'])
        df_comp['significant'] = df_comp['adjusted p-value'] < inp['pathway_alpha']
        df_ora = df_ora.append(df_comp)
    # Get logarithmic p-value
    df_ora['-log10(p-adjusted)'] = -1*numpy.log10(df_ora['adjusted p-value'])
    # Format output
    df_ora = df_ora.sort_values(['adjusted p-value','enrichment_factor'], ascending = True).reset_index(drop = True)
    # Save dataframe
    save_df(df_ora, inp['path_evaluation_ora'], f'pathway_ora_data_SN{inp["pre_signal_noise"]}{append_selection}', index = False)
    # Allocate data
    inp['pathway_ora'] = df_ora.copy()
    return inp

def scan_pathway_topology(inp):
    """
    Pathway topology analysis.

    Calculate pathway topology analysis and pathway impact for a comparison.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    """
    # Get results from ORA
    df_topology = inp['pathway_ora'].copy()
    # Network groups
    df_cluster = inp['data_cluster'].copy()
    # Get unique
    df_cluster = df_cluster[df_cluster['isobarics avoided'] == 'unique'].copy()
    # Get metabolite list
    df_filter = inp['uv_statistic_posthoc'].copy()
    # Get network data
    df_nodes = inp['pathway_analysis_data']['nodes'].copy()
    df_edges = inp['pathway_analysis_data']['edges'].copy()

    # Get all metabolites of organism
    metabolites_organism = set(df_edges['substrate'])|set(df_edges['product'])

    # Get all pathways of organism
    pathways_organism = set(df_edges['pathway_id'])

    # Get comparisons
    comparisons = inp['comparisons']

    # Select measure
    if inp['pathway_topology_measure'].lower() == 'closeness':
        key_measure = 'closeness_centrality'
        append_topo = '_closeness'
    elif inp['pathway_topology_measure'].lower() == 'load':
        key_measure = 'load_centrality'
        append_topo = '_load'
    elif inp['pathway_topology_measure'].lower() == 'harmonic':
        key_measure = 'harmonic_centrality'
        append_topo = '_harmonic'
    elif inp['pathway_topology_measure'].lower() == 'betweenness':
        key_measure = 'betweenness_centrality'
        append_topo = '_betweenness'
    else:
        key_measure = 'out-degree_centrality'
        append_topo = '_degree'

    # Cycle combinations
    for comparison in comparisons:
        # Qualitative
        df_cluster_qualitative_comparison = df_cluster[df_cluster['Group'].isin(list(comparison))].copy()
        df_cluster_qualitative_comparison_1 = df_cluster_qualitative_comparison[df_cluster_qualitative_comparison['Group'] == comparison[0]].copy()
        df_cluster_qualitative_comparison_2 = df_cluster_qualitative_comparison[df_cluster_qualitative_comparison['Group'] == comparison[1]].copy()
        set_qualitative_metabolites = set(df_cluster_qualitative_comparison['Component Name'])
        set_qualitative_metabolites_1 = set(df_cluster_qualitative_comparison_1['Component Name'])
        set_qualitative_metabolites_2 = set(df_cluster_qualitative_comparison_2['Component Name'])
        # Quantitative
        df_cluster_quantitative_comparison = df_cluster_qualitative_comparison[df_cluster_qualitative_comparison['quantitative'] == True].copy()
        df_cluster_quantitative_comparison_1 = df_cluster_qualitative_comparison[df_cluster_qualitative_comparison['Group'] == comparison[0]].copy()
        df_cluster_quantitative_comparison_2 = df_cluster_qualitative_comparison[df_cluster_qualitative_comparison['Group'] == comparison[1]].copy()
        set_quantitative_metabolites = set(df_cluster_quantitative_comparison['Component Name'])
        set_quantitative_metabolites_1 = set(df_cluster_quantitative_comparison_1['Component Name'])
        set_quantitative_metabolites_2 = set(df_cluster_quantitative_comparison_2['Component Name'])
        
        # Get statistic data for groups
        df_statistic_comparison = df_filter[(df_filter['group1'].isin(list(comparison)))&(df_filter['group2'].isin(list(comparison)))].copy()
        
        # Univariate selection list
        if inp['pathway_selection'] == 'univariate':
            df_comp = df_filter[df_filter['group1'].isin(list(comparison)) & df_filter['group2'].isin(list(comparison))].copy()
            df_comp = df_comp[df_comp['significant center']==True].copy()
            metabolites_significant = set(df_comp['Component Name'])
            append_selection = '_uni'
        # Multivariate selection list
        else:
            df_beta = inp['predictor_relevant_beta'].copy() 
            df_vip = inp['predictor_relevant_vip'].copy() 
            if len(comparisons) == 1:
                df_beta_selection = df_beta.copy()
                df_vip_selection = df_vip.copy()
            else:
                df_beta_selection = df_beta[df_beta['group'].isin(list(comparison))].copy()
                df_vip_selection = df_vip[df_vip['group'].isin(list(comparison))].copy()
            metabolites_significant = set(df_beta_selection['Component Name'])|set(df_vip_selection['Component Name'])
            append_selection = '_multi'
            
        # Nodes
        df_nodes_comparison = df_nodes.copy()
        # Set class
        df_nodes_comparison['class_1'] = comparison[0]
        df_nodes_comparison['class_2'] = comparison[1]
        # Qualitative  
        df_nodes_comparison['node_qualitative_class_1'] = numpy.where(df_nodes_comparison['compound_id'].isin(set_qualitative_metabolites_1), True, False)
        df_nodes_comparison['node_qualitative_class_2'] = numpy.where(df_nodes_comparison['compound_id'].isin(set_qualitative_metabolites_2), True, False)
        df_nodes_comparison['node_qualitative_classes_both'] = numpy.where((df_nodes_comparison['node_qualitative_class_1']==True)&(df_nodes_comparison['node_qualitative_class_2']==True), True, False)
        # Quantitative
        df_nodes_comparison['node_quantitative_class_1'] = numpy.where(df_nodes_comparison['compound_id'].isin(set_quantitative_metabolites_1), True, False)
        df_nodes_comparison['node_quantitative_class_2'] = numpy.where(df_nodes_comparison['compound_id'].isin(set_quantitative_metabolites_2), True, False)
        df_nodes_comparison['node_quantitative_classes_both'] = numpy.where((df_nodes_comparison['node_quantitative_class_1']==True)&(df_nodes_comparison['node_quantitative_class_2']==True), True, False)
        # Statistics
        df_nodes_comparison['significant']= numpy.where(df_nodes_comparison['compound_id'].isin(set(metabolites_significant)), True, False)
        df_nodes_comparison['upregulated'] = numpy.where(df_nodes_comparison['compound_id'].isin(set(df_statistic_comparison[df_statistic_comparison['FC']>1]['Component Name'])), True, False)
        df_nodes_comparison['downregulated'] = numpy.where(df_nodes_comparison['compound_id'].isin(set(df_statistic_comparison[df_statistic_comparison['FC']<1]['Component Name'])), True, False)
        df_nodes_comparison['significant_upregulated'] = numpy.where(df_nodes_comparison['significant'] & df_nodes_comparison['upregulated'], True, False)
        df_nodes_comparison['significant_downregulated'] = numpy.where(df_nodes_comparison['significant'] & df_nodes_comparison['downregulated'], True, False)

        # Edges
        df_edges_comparison = df_edges.copy()
        # Qualitative
        df_edges_comparison['edge_qualitative_class_1_both'] = numpy.where(
            ((df_edges_comparison['substrate'].isin(set_qualitative_metabolites_1))&
            (df_edges_comparison['product'].isin(set_qualitative_metabolites_1))),
            True, False
            )
        df_edges_comparison['edge_qualitative_class_2_both'] = numpy.where(
            ((df_edges_comparison['substrate'].isin(set_qualitative_metabolites_2))&
            (df_edges_comparison['product'].isin(set_qualitative_metabolites_2))),
            True, False
            )
        df_edges_comparison['edge_qualitative_classes_both'] = numpy.where(
            (df_edges_comparison['edge_qualitative_class_1_both']==True)&
            (df_edges_comparison['edge_qualitative_class_2_both']==True), 
            True, False
            )

        # Quantitative
        df_edges_comparison['edge_quantitative_class_1_both'] = numpy.where(
            ((df_edges_comparison['substrate'].isin(set_quantitative_metabolites_1))&
            (df_edges_comparison['product'].isin(set_quantitative_metabolites_1))),
            True, False
            )
        df_edges_comparison['edge_quantitative_class_2_both'] = numpy.where(
            ((df_edges_comparison['substrate'].isin(set_quantitative_metabolites_2))&
            (df_edges_comparison['product'].isin(set_quantitative_metabolites_2))),
            True, False
            )
        df_edges_comparison['edge_quantitative_classes_both'] = numpy.where(
            (df_edges_comparison['edge_quantitative_class_1_both']==True)&
            (df_edges_comparison['edge_quantitative_class_2_both']==True), 
            True, False
            )

        # Cycle pathways
        for pathway_id in pathways_organism:
            df_pathway = df_edges_comparison[df_edges_comparison['pathway_id'] == pathway_id].copy()
            metabolites_pathway = set(df_pathway['substrate'])|set(df_pathway['product'])
            metabolites_pathway_significant = metabolites_significant & metabolites_pathway
            
            # Create graph
            G = nx.DiGraph()

            G = nx.from_pandas_edgelist(
                df_pathway, 
                'substrate', 'product', 
                edge_attr = [
                    'organism_id', 'organism_name', 
                    'pathway_id', 'pathway_name', 
                    'reaction_id', 'type', 
                    'edge_qualitative_class_1_both', 'edge_qualitative_class_2_both', 'edge_qualitative_classes_both',
                    'edge_quantitative_class_1_both', 'edge_quantitative_class_2_both', 'edge_quantitative_classes_both'
                    ],
                create_using = G
                )
            
            # Nodes
            df_pathway_nodes = df_nodes_comparison.copy()
            df_pathway_nodes = df_pathway_nodes[df_pathway_nodes['compound_id'].isin(G.nodes)].copy()
            nx.set_node_attributes(G, df_pathway_nodes.set_index('compound_id').to_dict('index'))
            nx.set_node_attributes(G, {item: item for item in G.nodes()}, name = 'compound_id')

            # Network analysis
            subgraph = G
            subgraph = scan_network_analysis_main(subgraph)

            # Create dictionary with information
            dict_network = scan_network_analysis_format(subgraph)
            filename = f'network_analysis_topology_comparison_full_SN{inp["pre_signal_noise"]}_{pathway_id}'
            save_dict(dict_network, inp['path_evaluation_topology'], filename = filename, single_files = False, index = False)
            scan_network_save_cytoscape(subgraph, inp['path_evaluation_topology'], filename = filename) 

            # Calculate pathway impact
            impact = numpy.sum([subgraph.nodes[item][key_measure] for item in metabolites_pathway_significant]) / numpy.sum([subgraph.nodes[item][key_measure] for item in metabolites_pathway])

            index_ora = numpy.where((df_topology['group1'].isin(list(comparison)))&(df_topology['group2'].isin(list(comparison))&(df_topology['pathway_id'] == pathway_id)))[0]
            df_topology.at[index_ora, 'pathway_impact'] = impact

    # Save dataframe
    save_df(df_topology, inp['path_evaluation_topology'], f'pathway_topology_data_SN{inp["pre_signal_noise"]}{append_selection}{append_topo}', index = False)
    # Allocate data
    inp['pathway_topology'] = df_topology.copy()
    return inp

def scan_pathway_msea(inp):
    """
    Metabolite set enrichment analysis.

    Calculate Goemans global test for Metabolite set enrichment analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.

    Returns
    -------
    inp : dict
        Method dictionary.
    """

    # Allocate data
    df_filter = inp['data_processed_filter'].copy()
    
    # Exclude QC
    df_filter = df_filter[~df_filter['Group'].str.contains('QC ')].copy()
    df_filter = df_filter.sort_values(['Group']).reset_index(drop = True)

    # Get network data
    df_nodes = inp['pathway_analysis_data']['nodes'].copy()
    df_edges = inp['pathway_analysis_data']['edges'].copy()

    # Get all metabolites of organism
    metabolites_organism = set(df_edges['substrate'])|set(df_edges['product'])
    
    # Get all pathways of organism
    pathways_organism = set(df_edges['pathway_id'])
    
    # Get comparisons
    comparisons = inp['comparisons']

    # Formatting
    df_scaled = scan_mv_preprocessing(df_filter)
    # Standardization, auto-scaling
    X_scale = df_scaled[df_scaled.columns[~df_scaled.columns.isin(['Component Name','Group'])]].values
    df_scaled[df_scaled.columns[~df_scaled.columns.isin(['Component Name','Group'])]] = scan_pathway_preprocessing_scaling(X_scale)
    
    # Preallocate
    list_global_test = []
    # Cycle combinations
    for comparison in comparisons:
        df_comp = df_scaled[df_scaled['Group'].isin(list(comparison))].copy()
        mapper_group = dict([(item, i) for (i, item) in enumerate(df_comp['Group'].unique())])
        df_comp['Group'] = df_comp['Group'].map(mapper_group)
        metabolites_significant = set(df_scaled.columns[~df_scaled.columns.isin(['Component Name','Group'])])

        # Cycle pathways
        for pathway_id in pathways_organism:
            df_pathway = df_edges[df_edges['pathway_id'] == pathway_id].copy()
            metabolites_pathway = set(df_pathway['substrate'])|set(df_pathway['product'])
            metabolites_pathway_significant = metabolites_significant & metabolites_pathway

            # Allocate variables
            N = len(metabolites_organism)
            K = len(metabolites_significant)

            n = len(metabolites_pathway)
            k = len(metabolites_pathway_significant)

            if (n >= 2)&(k >= 1):
                df_X_pathway = df_comp[df_comp.columns[df_comp.columns.isin(metabolites_pathway_significant)]].copy()
                df_y_pathway = df_comp[df_comp.columns[df_comp.columns.isin(['Group'])]].copy()

                X = df_X_pathway.values
                y = df_y_pathway.values
                
                # Get expected Q statistic
                Q_expected = numpy.trace(scan_pathway_msea_R(X))

                # Calculate Q statistic
                Q = scan_pathway_qstat(X,y)

                # Permutation test
                p = scan_pathway_permutation_p(X,y,Q)

                # Get enrichment factor
                ef = Q/Q_expected
                
                # Append test
                list_global_test.append(
                    [
                        comparison[0], comparison[1], 
                        pathway_id, inp['mapper_pathway'][pathway_id], 
                        N, K, 
                        n, k,  
                        Q, Q_expected, ef, p,
                        len(df_y_pathway), len(df_y_pathway[df_y_pathway['Group']==0].copy()), len(df_y_pathway[df_y_pathway['Group']==1].copy()),
                    ]
                    )

    # Create dataframe
    df_msea_hold = pandas.DataFrame(
        list_global_test, 
        columns = [
            'group1', 'group2', 
            'pathway_id', 'pathway_name',
            'metabolites_organism', 'metabolites_significant', 
            'metabolites_pathway', 'metabolites_pathway_significant',
            'Q statistic', 'Q expected', 'enrichment_factor', 'p-value',
            'sample_number_all', 'sample_number_group1', 'sample_number_group2',
            ]
        )
    
    # Multi-comparison correction
    df_msea = pandas.DataFrame()
    # Cycle comparisons
    for comparison in comparisons:
        df_comp = df_msea_hold[df_msea_hold['group1'].isin(list(comparison)) & df_msea_hold['group2'].isin(list(comparison))].copy()
        df_comp['adjusted p-value'] = multiple_testing_correction(pvalues = list(df_comp['p-value']), correction_type=inp['pathway_correction'])
        df_comp['significant'] = df_comp['adjusted p-value'] < inp['pathway_alpha']
        df_msea = df_msea.append(df_comp)
    # Get logarithmic p-value
    df_msea['-log10(p-adjusted)'] = -1*numpy.log10(df_msea['adjusted p-value'])
    # Format output
    df_msea = df_msea.sort_values(['adjusted p-value'], ascending = True)
    # Save dataframe
    save_df(df_msea, inp['path_evaluation_msea'], f'pathway_msea_data_SN{inp["pre_signal_noise"]}_auto', index = False)
    # Allocate data
    inp['pathway_msea'] = df_msea.copy()
    return inp

def scan_pathway_preprocessing_scaling(X):
    """
    Scaling.

    Fixed autoscaling for pathway analysis.

    Parameters
    ----------
    X : array
        Unscaled pivot data.

    Returns
    -------
    X_new : array
        Scaled pivot data.
    """
    scaler = CustomScalerAuto()
    X_new = scaler.fit_transform(X)
    return X_new

def scan_pathway_qstat(X,y):
    """
    Q-Statistic.

    Calculate Q-Statistic for global scan.

    Parameters
    ----------
    X : array
        Pivot data.
    y : array
        Response.

    Returns
    -------
    Q : float
        Q-statistic.
    """
    # Calculate expectation of Y
    q = len(numpy.where(y==1)[0])
    t = len(y)
    mu = q/t
    # Mean centering Y
    Z = y - mu
    # Calculate matrix R (configuration matrix of the samples)
    R = scan_pathway_msea_R(X)
    # Calculate Q statistic
    Q = (Z.T@R@Z)/(mu*(1-mu))
    return Q.squeeze()

def scan_pathway_permutation_p(X,y,Q):
    """
    Permutation test fixed.

    Run permutation test with fixed permutations for large sample sizes.

    Parameters
    ----------
    X : array
        Pivot data.
    y : array
        Response.
    Q : float
        Q-statistic.

    Returns
    -------
    p : float
        Probability.
    """
    # Set number of permutations
    n_permutations = 10000
    # Cycle permuations
    k = 1
    for i in range(n_permutations):
        # Perform permutation of the output
        ind_permutation = numpy.random.permutation(len(y))
        # Permuted output vector
        y_i = y[ind_permutation]
        # Q statistic for permuted output vector
        Q_i = scan_pathway_qstat(X,y_i)
        # Comparing Q of real response with Q of permutation
        if Q_i >= Q:
            k+=1
    # Calculate p-value
    p = k/n_permutations
    return p

def scan_pathway_msea_R(X):
    """
    Get R matrix.

    Get configuration matrix of X

    Parameters
    ----------
    X : array
        Pivot data.
    
    Returns
    -------
    R : array
        Configuration matrix.
    """
    # Get number of features
    m = X.shape[1]
    # Calculate matrix R (configuration matrix of the samples)
    R = (1/m)*(X@X.T)
    return R

def scan_pathway_ora_plot(inp):
    """
    Plot over representation analysis.

    Plotting function for over representation analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    """
    # Get data
    df = inp['pathway_ora'].copy()
    # Get comparisons
    comparisons = inp['comparisons'].copy()
    # Cycle comparisons
    for comparison in comparisons:
        df_comparison = df[(df['group1'].isin(list(comparison)))&(df['group2'].isin(list(comparison)))].copy()
        df_comparison = df_comparison.sort_values(['-log10(p-adjusted)'], ascending = False).reset_index(drop = True)
        
        # Only show top candidates, double sorting due to YX plotting.
        df_comparison = df_comparison[:inp['pathway_number_top']]
        df_comparison = df_comparison.sort_values(['-log10(p-adjusted)','enrichment_factor'], ascending = True).reset_index(drop = True)
        
        # Legend properties
        legendproperties = {'size': inp['pathway_labelsize'], 'weight': 'bold'}

        # Create plot   
        fig = plt.figure(figsize = inp['pathway_figsize'])
        # Add plots for multiple axes
        ax1=fig.add_subplot(111, label="1", frameon = True)
        ax2=fig.add_subplot(111, label="2", frameon = False)
        ax3=fig.add_subplot(111, label="3", frameon = False)
        ax2.get_xaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax2.yaxis.tick_right()
        
        # Remove axis
        ax2.tick_params(left = False, right = True, labelleft = False, labelright = True)
        ax3.tick_params(left = False, right = False, labelleft = False, labelright = False)

        # Plot data
        seaborn.scatterplot(
            data = df_comparison, 
            x = 'enrichment_factor', y = 'pathway_name', 
            hue = '-log10(p-adjusted)', palette = 'YlOrRd',
            edgecolor = 'k', linewidth = 0.8, linestyle = '-', s = 100,
            legend = 'brief', ax = ax1)

        # Set ticks
        ax1.yaxis.set_ticks(df_comparison.index)
        ax2.yaxis.set_ticks(df_comparison.index)

        # Scale axes
        ax1.set_xlim(0)
        ax1.set_ylim(numpy.nanmin(ax1.get_yticks())-0.5, numpy.nanmax(ax1.get_yticks())+0.5)
        ax2.set_ylim(numpy.nanmin(ax1.get_yticks())-0.5, numpy.nanmax(ax1.get_yticks())+0.5)

        # Format
        ax1.tick_params(axis = 'both', labelsize = inp['pathway_labelsize'])
        ax2.tick_params(axis = 'both', labelsize = inp['pathway_labelsize'])
        ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        marker_significant = u'\u2605'
        ax2.set_yticklabels([f'$\it{marker_significant}$' if item == True else '' for item in df_comparison['significant']], ha = 'left', va = 'center', fontweight = 'bold')
        ax1.set_xlabel('Enrichment factor', fontsize = inp['pathway_labelsize'], fontweight = 'bold')
        ax1.set_ylabel('')

        # Draw grid
        ax1.set_axisbelow(True)
        ax1.yaxis.grid()
        plt.draw()

        if inp['pathway_selection'] == 'univariate':
            append_selection = '_uni'
        else:
            append_selection = '_multi'

        # Create legend              
        list_bbox = [
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'alpha = {inp["pathway_alpha"]}'), 
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'correction = {inp["pathway_correction"]}'),
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'list = {inp["pathway_selection"]}')
            ]
        list_combo = [
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'{comparison[0]} over {comparison[1]}')
        ]
        legendproperties = {'size': inp['pathway_labelsize'], 'weight': 'bold'}
        leg1 = ax1.legend(bbox_to_anchor=(0.0, 1.0), loc="lower right", frameon = False, prop = legendproperties, title = '-log$_{10}$(p$_{adjusted}$)', title_fontsize = inp['pathway_labelsize'])
        leg2 = ax2.legend(handles = list_bbox, bbox_to_anchor=(0.0, 1.0), handlelength = 0,  loc="lower left", frameon = False, prop = legendproperties)
        leg3 = ax3.legend(handles = list_combo, bbox_to_anchor=(1.0, 1.0), handlelength = 0,  loc="lower right", frameon = False, prop = legendproperties)

        # Save plot
        list_bbox1 = [leg1, leg2, leg3]
        fig.savefig(inp['path_evaluation_ora'].joinpath(f'pathway_ora_{comparison}_SN{inp["pre_signal_noise"]}{append_selection}.png'), bbox_extra_artists = (list_bbox1), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_ora'].joinpath(f'pathway_ora_{comparison}_SN{inp["pre_signal_noise"]}{append_selection}.svg'), format='svg' , bbox_extra_artists = (list_bbox1), bbox_inches = 'tight', dpi = 800)
        plt.close(fig)
        
    return

def scan_pathway_topology_plot(inp):
    """
    Plot topology analysis.

    Plotting function for topology analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    """
    # Get data
    df = inp['pathway_topology'].copy()
    # Get comparisons
    comparisons = inp['comparisons'].copy()
    # Cycle comparisons
    for comparison in comparisons:
        # Select comparison
        df_comparison = df[(df['group1'].isin(list(comparison)))&(df['group2'].isin(list(comparison)))].copy()
        df_comparison = df_comparison.sort_values(['-log10(p-adjusted)'], ascending = False).reset_index(drop = True)
        
        # Only show top candidates, double sorting due to YX plotting.
        df_comparison = df_comparison[:inp['pathway_number_top']]
        df_comparison = df_comparison.sort_values(['-log10(p-adjusted)','enrichment_factor'], ascending = True).reset_index(drop = True)
        
        # Legend properties
        legendproperties = {'size': inp['pathway_labelsize'], 'weight': 'bold'}

        # Create plot   
        fig = plt.figure(figsize = inp['pathway_figsize'])
        # Add plots for multiple axes
        ax1=fig.add_subplot(111, label="1", frameon = True)
        ax2=fig.add_subplot(111, label="2", frameon = False)
        ax3=fig.add_subplot(111, label="3", frameon = False)
        ax2.get_xaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax2.yaxis.tick_right()
        
        # Remove axis
        ax2.tick_params(left = False, right = True, labelleft = False, labelright = True)
        ax3.tick_params(left = False, right = False, labelleft = False, labelright = False)
        
        # Plot data
        seaborn.scatterplot(
            data = df_comparison, 
            x = 'pathway_impact', y = 'pathway_name', 
            hue = '-log10(p-adjusted)', palette = 'YlOrRd',
            edgecolor = 'k', linewidth = 0.8, linestyle = '-', s = 100,
            legend = 'brief', ax = ax1)

        # Set ticks
        ax1.set_xticks([0,0.2,0.4,0.6,0.8,1])
        ax1.yaxis.set_ticks(df_comparison.index)
        ax2.yaxis.set_ticks(df_comparison.index)

        # Scale axes
        ax1.set_xlim(-0.1,1.1)
        ax1.set_ylim(numpy.nanmin(ax1.get_yticks())-0.5, numpy.nanmax(ax1.get_yticks())+0.5)
        ax2.set_ylim(numpy.nanmin(ax1.get_yticks())-0.5, numpy.nanmax(ax1.get_yticks())+0.5)

        # Format
        ax1.tick_params(axis = 'both', labelsize = inp['pathway_labelsize'])
        ax2.tick_params(axis = 'both', labelsize = inp['pathway_labelsize'])
        ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        marker_significant = u'\u2605'
        ax2.set_yticklabels([f'$\it{marker_significant}$' if item == True else '' for item in df_comparison['significant']], ha = 'left', va = 'center', fontweight = 'bold')
        ax1.set_xlabel('Pathway impact', fontsize = inp['pathway_labelsize'], fontweight = 'bold')
        ax1.set_ylabel('')

        # Select measure
        if inp['pathway_topology_measure'].lower() == 'closeness':
            append_topo = '_closeness'
        elif inp['pathway_topology_measure'].lower() == 'load':
            append_topo = '_load'
        elif inp['pathway_topology_measure'].lower() == 'harmonic':
            append_topo = '_harmonic'
        elif inp['pathway_topology_measure'].lower() == 'betweenness':
            append_topo = '_betweenness'
        else:
            append_topo = '_degree'

        # Draw grid
        ax1.set_axisbelow(True)
        ax1.yaxis.grid()
        plt.draw()

        if inp['pathway_selection'] == 'univariate':
            append_selection = '_uni'
        else:
            append_selection = '_multi'

        # Create legend              
        list_bbox = [
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'alpha = {inp["pathway_alpha"]}'), 
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'correction = {inp["pathway_correction"]}'),
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'list = {inp["pathway_selection"]}'),
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'centrality = {inp["pathway_topology_measure"]}')
            ]
        list_combo = [
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'{comparison[0]} over {comparison[1]}')
        ]
        legendproperties = {'size': inp['pathway_labelsize'], 'weight': 'bold'}
        leg1 = ax1.legend(bbox_to_anchor=(0.0, 1.0), loc="lower right", frameon = False, prop = legendproperties, title = '-log$_{10}$(p$_{adjusted}$)', title_fontsize = inp['pathway_labelsize'])
        leg2 = ax2.legend(handles = list_bbox, bbox_to_anchor=(0.0, 1.0), handlelength = 0,  loc="lower left", frameon = False, prop = legendproperties)
        leg3 = ax3.legend(handles = list_combo, bbox_to_anchor=(1.0, 1.0), handlelength = 0,  loc="lower right", frameon = False, prop = legendproperties)

        # Save plot
        list_bbox1 = [leg1, leg2, leg3]
        fig.savefig(inp['path_evaluation_topology'].joinpath(f'pathway_topology_{comparison}_SN{inp["pre_signal_noise"]}{append_selection}{append_topo}.png'), bbox_extra_artists = (list_bbox1), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_topology'].joinpath(f'pathway_topology_{comparison}_SN{inp["pre_signal_noise"]}{append_selection}{append_topo}.svg'), format='svg' , bbox_extra_artists = (list_bbox1), bbox_inches = 'tight', dpi = 800)
        plt.close(fig)
        
    return

def scan_pathway_msea_plot(inp):
    """
    Plot metabolite set enrichment analysis.

    Plotting function for metabolite set enrichment analysis.

    Parameters
    ----------
    inp : dict
        Method dictionary.
    """
    # Get data
    df = inp['pathway_msea'].copy()
    # Get comparisons
    comparisons = inp['comparisons'].copy()
    # Cycle comparisons
    for comparison in comparisons:
        df_comparison = df[(df['group1'].isin(list(comparison)))&(df['group2'].isin(list(comparison)))].copy()
        df_comparison = df_comparison.sort_values(['-log10(p-adjusted)'], ascending = False).reset_index(drop = True)
        df_comparison['Q statistic'] = df_comparison['Q statistic'].apply(lambda x: float(x))
        
        # Only show top candidates, double sorting due to YX plotting.
        df_comparison = df_comparison[:inp['pathway_number_top']]
        df_comparison = df_comparison.sort_values(['-log10(p-adjusted)','enrichment_factor'], ascending = True).reset_index(drop = True)
        
        # Legend properties
        legendproperties = {'size': inp['pathway_labelsize'], 'weight': 'bold'}

        # Create plot   
        fig = plt.figure(figsize = inp['pathway_figsize'])
        # Add plots for multiple axes
        ax1=fig.add_subplot(111, label="1", frameon = True)
        ax2=fig.add_subplot(111, label="2", frameon = False)
        ax3=fig.add_subplot(111, label="3", frameon = False)
        ax2.get_xaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax2.yaxis.tick_right()
        
        # Remove axis
        ax2.tick_params(left = False, right = True, labelleft = False, labelright = True)
        ax3.tick_params(left = False, right = False, labelleft = False, labelright = False)

        # Plot data
        seaborn.scatterplot(
            data = df_comparison, 
            x = 'enrichment_factor', y = 'pathway_name', 
            hue = '-log10(p-adjusted)', palette = 'YlOrRd',
            size = 'Q statistic', sizes = (20,200), edgecolor = 'k', linewidth = 0.8, linestyle = '-',
            legend = 'brief', ax = ax1)

        # Set ticks
        ax1.yaxis.set_ticks(df_comparison.index)
        ax2.yaxis.set_ticks(df_comparison.index)

        # Scale axes
        ax1.set_xlim(0)
        ax1.set_ylim(numpy.nanmin(ax1.get_yticks())-0.5, numpy.nanmax(ax1.get_yticks())+0.5)
        ax2.set_ylim(numpy.nanmin(ax1.get_yticks())-0.5, numpy.nanmax(ax1.get_yticks())+0.5)

        # Format
        ax1.tick_params(axis = 'both', labelsize = inp['pathway_labelsize'])
        ax2.tick_params(axis = 'both', labelsize = inp['pathway_labelsize'])
        ax1.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        ax2.tick_params(axis = 'y', length = 0, labelsize = inp['mv_labelsize'])
        marker_significant = u'\u2605'
        ax2.set_yticklabels([f'$\it{marker_significant}$' if item == True else '' for item in df_comparison['significant']], ha = 'left', va = 'center', fontweight = 'bold')
        ax1.set_xlabel('Enrichment factor', fontsize = inp['pathway_labelsize'], fontweight = 'bold')
        ax1.set_ylabel('')

        # Draw grid
        ax1.set_axisbelow(True)
        ax1.yaxis.grid()
        plt.draw()

        # Create legend              
        list_bbox = [
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'alpha = {inp["pathway_alpha"]}'), 
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'correction = {inp["pathway_correction"]}')
            ]
        list_combo = [
            mlines.Line2D([],[], marker=None, linewidth = 0, label = f'{comparison[0]} and {comparison[1]}')
        ]
        legendproperties = {'size': inp['pathway_labelsize'], 'weight': 'bold'}
        leg1 = ax1.legend(bbox_to_anchor=(0.0, 1.0), loc="lower right", frameon = False, prop = legendproperties, title = 'Q statistic', title_fontsize = inp['pathway_labelsize'])
        leg2 = ax2.legend(handles = list_bbox, bbox_to_anchor=(0.0, 1.0), handlelength = 0,  loc="lower left", frameon = False, prop = legendproperties)
        leg3 = ax3.legend(handles = list_combo, bbox_to_anchor=(1.0, 1.0), handlelength = 0,  loc="lower right", frameon = False, prop = legendproperties)
        
        # Save plot
        list_bbox1 = [leg1, leg2, leg3]
        fig.savefig(inp['path_evaluation_msea'].joinpath(f'pathway_msea_{comparison}_SN{inp["pre_signal_noise"]}_auto.png'), bbox_extra_artists = (list_bbox1), bbox_inches = 'tight', dpi = 800)
        fig.savefig(inp['path_evaluation_msea'].joinpath(f'pathway_msea_{comparison}_SN{inp["pre_signal_noise"]}_auto.svg'), format='svg' , bbox_extra_artists = (list_bbox1), bbox_inches = 'tight', dpi = 800)
        plt.close(fig)
    return

def scan_network_save_cytoscape(network, path_folder, filename):
    """
    Comparison network analysis.

    Calculate network analysis for a comparison. Plotting is done with json files in Cytoscape.

    Parameters
    ----------
    network : object
        Network object to save.
    path_folder : path
        Path to results folder.
    filename : str
        Filename.
    """
    cs = nx.readwrite.json_graph.cytoscape_data(network)
    # the json file where the output must be stored 
    path_file = path_folder.joinpath(f'{filename}.json')
    out_file = open(f'{path_file}', "w")   
    json.dump(cs, out_file, indent = 6) 
    out_file.close() 
    return

def scan_network_analysis_main(network_empty):
    """
    Network analysis.

    Calculate network metrics.

    Parameters
    ----------
    network_empty : object
        Network after creation without metrics.

    Returns
    -------
    network : object
        Network with metrics.
    """
    # Create copy
    network = network_empty.copy()
 
    # Class
    network.graph['class'] = {}
    network.graph['class']['is_directed'] = network.is_directed()
    network.graph['class']['is_multigraph'] = network.is_multigraph()
    # Nodes
    network.graph['nodes'] = {}
    network.graph['nodes']['number_of_nodes'] = nx.number_of_nodes(network)
    # Edges
    network.graph['edges'] = {}
    network.graph['edges']['number_of_edges'] = nx.number_of_edges(network)
    network.graph['edges']['density'] = nx.density(network)
    # Selfloops
    network.graph['selfloops'] = {}
    network.graph['selfloops']['number_of_selfloops'] = nx.number_of_selfloops(network)
    # Connectivity
    if network.graph['class']['is_directed'] == False:
        network.graph['connectivity'] = {}
        network.graph['connectivity']['is_connected'] = nx.is_connected(network)
        network.graph['connectivity']['number_connected_components'] = nx.number_connected_components(network)

    # Assortativity
    network = scan_network_analysis_set_node_attribute(network, nx.average_neighbor_degree(network), 'average_neighbor_degree')
    if network.graph['class']['is_directed'] == False:
        if network.graph['connectivity']['is_connected'] == True:
            network.graph['assortativity'] = {}
            network.graph['assortativity']['degree_assortativity_coefficient'] =  nx.degree_assortativity_coefficient(network)
            network.graph['assortativity']['degree_pearson_correlation_coefficient'] = nx.degree_pearson_correlation_coefficient(network)
    
    # Bridges
    if network.graph['class']['is_multigraph'] == False:
        if network.graph['class']['is_directed'] == False:
            network.graph['bridges'] = {}
            network.graph['bridges']['has_bridges'] = nx.has_bridges(network)
    
    # Centrality
    network = scan_network_analysis_set_node_attribute(network, {i:j for i,j in network.degree()}, 'degree')
    network = scan_network_analysis_set_node_attribute(network, nx.closeness_centrality(network), 'closeness_centrality')
    network = scan_network_analysis_set_node_attribute(network, nx.load_centrality(network), 'load_centrality')
    network = scan_network_analysis_set_node_attribute(network, nx.harmonic_centrality(network), 'harmonic_centrality')
    if network.graph['class']['is_directed'] == True:
        network = scan_network_analysis_set_node_attribute(network, {i:j for i,j in network.in_degree()}, 'in-degree')
        network = scan_network_analysis_set_node_attribute(network, {i:j for i,j in network.out_degree()}, 'out-degree')
        network = scan_network_analysis_set_node_attribute(network, nx.in_degree_centrality(network), 'in-degree_centrality')
        network = scan_network_analysis_set_node_attribute(network, nx.out_degree_centrality(network), 'out-degree_centrality')

    if network.graph['class']['is_multigraph'] == False:
        network = scan_network_analysis_set_node_attribute(network, nx.degree_centrality(network), 'degree_centrality')            
        #network = scan_network_analysis_set_node_attribute(network, nx.eigenvector_centrality(network), 'eigenvector_centrality')
        network = scan_network_analysis_set_node_attribute(network, nx.betweenness_centrality(network), 'betweenness_centrality')
        network = scan_network_analysis_set_edge_attribute(network, nx.edge_betweenness_centrality(network), 'edge_betweenness_centrality')
        #network = scan_network_analysis_set_edge_attribute(network, nx.edge_load_centrality(network), 'edge_load_centrality')

    if network.graph['class']['is_directed'] == False:
        if network.graph['connectivity']['is_connected'] == True:
            second_order_centrality = nx.second_order_centrality(network)
            network = scan_network_analysis_set_node_attribute(network, second_order_centrality, 'second_order_centrality')
    
    set_voterank = nx.voterank(network)
    voterank = {key: value for (key, value) in [(item, True) if item in set_voterank else (item, False) for item in network.nodes]}
    network = scan_network_analysis_set_node_attribute(network, voterank, 'voterank')
    
    # Clustering
    network = scan_network_analysis_set_node_attribute(network, nx.square_clustering(network), 'square_clustering')
    if network.graph['class']['is_multigraph'] == False:
        network.graph['clustering'] = {}
        network.graph['clustering']['transitivity'] = nx.transitivity(network)
        network.graph['clustering']['average_clustering'] = nx.average_clustering(network)
        network = scan_network_analysis_set_node_attribute(network, nx.clustering(network), 'clustering')
        if network.graph['class']['is_directed'] == False:
            network = scan_network_analysis_set_node_attribute(network, nx.triangles(network), 'triangles')
    
    # Communities
    if network.graph['class']['is_directed'] == False:
        communities = nx.algorithms.community.greedy_modularity_communities(network)
        for community_number, community in enumerate(communities):
            nx.set_node_attributes(network, name='community', values={i:community_number for i in community})
        
    # Distance
    if network.graph['class']['is_directed'] == False:
        if network.graph['connectivity']['is_connected'] == True:
            network.graph['distance'] = {}
            network.graph['distance']['diameter'] = nx.diameter(network)
            network.graph['distance']['extrema_bounding'] = nx.extrema_bounding(network)
            network.graph['distance']['radius'] = nx.radius(network)

            set_center = nx.center(network)
            center = {key: value for (key, value) in [(item, True) if item in set_center else (item, False) for item in network.nodes]}
            network = scan_network_analysis_set_node_attribute(network, center, 'center')

            set_barycenter = nx.barycenter(network)
            barycenter = {key: value for (key, value) in [(item, True) if item in set_barycenter else (item, False) for item in network.nodes]}
            network = scan_network_analysis_set_node_attribute(network, barycenter, 'barycenter')

            set_periphery = nx.periphery(network)
            periphery = {key: value for (key, value) in [(item, True) if item in set_periphery else (item, False) for item in network.nodes]}
            network = scan_network_analysis_set_node_attribute(network, periphery, 'periphery')

            eccentricity = nx.eccentricity(network)
            network = scan_network_analysis_set_node_attribute(network, eccentricity, 'eccentricity')
    
    if (network.graph['class']['is_directed'] == False)&(network.graph['class']['is_multigraph'] == False):
        network.graph['distance']['is_distance_regular'] = nx.is_distance_regular(network)
        network.graph['distance']['is_strongly_regular'] = nx.is_strongly_regular(network)
        if network.graph['distance']['is_distance_regular'] == True:
            network.graph['distance']['intersection_array'] = nx.intersection_array(network)

    # Dominance
    set_dominance = nx.dominating_set(network)
    dominating_set = {key: value for (key, value) in [(item, True) if item in set_dominance else (item, False) for item in network.nodes]}
    network = scan_network_analysis_set_node_attribute(network, dominating_set, 'dominating_set')
    # Efficiency
    if network.graph['class']['is_directed'] == False:
        network.graph['efficiency'] = {}
        network.graph['efficiency']['local_efficiency'] = nx.local_efficiency(network)
        network.graph['efficiency']['global_efficiency'] = nx.global_efficiency(network)
    # Isolates
    set_isolate = nx.isolates(network)
    isolate = {key: value for (key, value) in [(item, True) if item in set_isolate else (item, False) for item in network.nodes]}
    network = scan_network_analysis_set_node_attribute(network, isolate, 'isolate')
    # Link analysis
    if network.graph['class']['is_directed'] == False:
        if (network.graph['connectivity']['is_connected'] == True)&(network.graph['class']['is_multigraph'] == False):
            pagerank = {key: value for (key, value) in [(item,i) for (i,item) in enumerate(nx.pagerank(network), start = 1)]}
            network = scan_network_analysis_set_node_attribute(network, pagerank, 'pagerank')

        if network.graph['class']['is_multigraph'] == False:
            hits, authorities = nx.hits(network)
            network = scan_network_analysis_set_node_attribute(network, hits, 'hits')
            network = scan_network_analysis_set_node_attribute(network, authorities, 'authorities')
    # Matching
    if network.graph['class']['is_directed'] == False:
        set_matching = nx.maximal_matching(network)
        maximal_matching = {key: value for (key, value) in [(item, True) if item in set_matching else (item, False) for item in network.edges]}
        network = scan_network_analysis_set_edge_attribute(network, maximal_matching, 'maximal_matching')
    # s metric
    network.graph['s_metric'] = {}
    network.graph['s_metric']['s_metric'] = nx.s_metric(network, normalized = False)
    # Structural holes
    network = scan_network_analysis_set_node_attribute(network, nx.constraint(network), 'constraint')
    network = scan_network_analysis_set_node_attribute(network, nx.effective_size(network), 'effective_size')
    return network

def scan_network_analysis_set_node_attribute(network, dictionary, attribute_name):
    """
    Set node attribute.

    Set node attribute with metric dictionary.

    Parameters
    ----------
    network : object
        Network object.
    dictionary : dict
        Dictionary of metric.
    attribute_name : str
        Name for attribute.

    Returns
    -------
    network : object
        Network with metrics.
    """
    for key in dictionary.keys():
        if not pandas.isna(dictionary[key]):
            network.nodes[key][attribute_name] = dictionary[key]
        else:
            network.nodes[key][attribute_name] = 0
    return network

def scan_network_analysis_set_edge_attribute(network, dictionary, attribute_name):
    """
    Set edge attribute.

    Set edge attribute with metric dictionary.

    Parameters
    ----------
    network : object
        Network object.
    dictionary : dict
        Dictionary of metric.
    attribute_name : str
        Name for attribute.

    Returns
    -------
    network : object
        Network with metrics.
    """
    for key in dictionary.keys():
        if not pandas.isna(dictionary[key]):
            network.edges[key][attribute_name] = dictionary[key]
        else:
            network.edges[key][attribute_name] = 0
    return network

def scan_network_analysis_format(network):
    """
    Format network.

    Create network dictionary for data storage.

    Parameters
    ----------
    network : object
        Network object.

    Returns
    -------
    dict_network : dict
        Dictionary for network.
    """
    list_common = []
    for key_class in network.graph.keys():
        dict_class = network.graph[key_class].copy()
        for key_variable in dict_class.keys():
            list_common.append([key_class, key_variable, dict_class[key_variable]])
    df_common = pandas.DataFrame(list_common, columns = ['class','variable','value'])

    df_nodes = pandas.DataFrame()
    for i, key_node in enumerate(network.nodes):
        df_nodes.at[i, 'node'] = key_node
        dict_node = network.nodes[key_node].copy()
        for key_variable in [item for item in dict_node.keys() if 'color' not in item]:
            df_nodes.at[i, key_variable] = dict_node[key_variable]

    df_edges = pandas.DataFrame()
    for i, key_edge in enumerate(network.edges):
        df_edges.at[i, 'source'] = key_edge[0]
        df_edges.at[i, 'target'] = key_edge[1]
        dict_edge = network.edges[key_edge].copy()
        for key_variable in [item for item in dict_edge.keys() if 'color' not in item]:
            df_edges.at[i, key_variable] = dict_edge[key_variable]

    dict_network = {}
    dict_network['information'] = df_common
    dict_network['nodes'] = df_nodes
    dict_network['edges'] = df_edges
    return dict_network

def scan_expand_convolution(df, inp):
    """
    Expand covoluted names.

    Parameter
    ---------
    df : dataframe
        Dataframe with convoluted names.
    inp : dict
        Method dictionary.

    Returns
    -------
    df : dataframe
        Dataframe with single names.
    """
    # Split component name lists of convolutions
    df = splitDataFrameList(df, 'Component Name', '_').copy()
    # Map component name
    df['compound_name'] = df['Component Name'].map(inp['mapper_name'])
    df = df.rename(columns = {'Component Name': 'compound_id'})
    return df

def scan_get_single_masstransitions(df):
    """
    Select single mass trainsitions.

    Parameter
    ---------
    df : dataframe
        Dataframe with convoluted mass transitions.

    Returns
    -------
    df : dataframe
        Dataframe with single mass transitions.
    """
    list_single_data = [item[0] for item in df['Component Name'].str.split('_') if len(item) == 1]
    df = df[df['Component Name'].isin(set(list_single_data))].copy()
    return df