import torch

def ridge_regression(A, B, alpha):
    # Calculate the regularization matrix to solve the ill posed problem of excessively large condition numbers.
    b_size = A.shape[0] # batch size
    reg = alpha * torch.eye(A.shape[2]).to(A.device).unsqueeze(0).repeat(b_size, 1, 1)
    # If m≥n, lstsq() solves the least-squares problem; else, lstsq() solves the least-norm problem
    # Returned tensor X has shape (max(m,n)×k). so it is better to make n >= m
    # need pytorch version >= 1.10.0
    X = torch.linalg.lstsq(torch.matmul(A.transpose(1, 2), A) + reg, torch.matmul(A.transpose(1, 2), B)).solution #K = (A^T A + λE)^-1 A^T B, λE = reg = alpha * I (alpha applied once)
    return X