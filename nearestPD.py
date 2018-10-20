import torch
from numpy import linalg as la





























def isPD(B):
	try:
		M = B.detach().numpy()
		_ = la.cholesky(M)
		return True
	except la.LinAlgError:
	return False