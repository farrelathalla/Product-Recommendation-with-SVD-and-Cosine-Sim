import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class RecommendationSystem:
    def __init__(self, k_factors=4):
        self.k_factors = k_factors
        
    def create_sample_data(self):
        "Sample User-Item Matrix"
        ratings = [
        [3.0, 4.0, 5.0, 0.0, 0.0, 5.0, 4.0, 0.0, 3.0, 3.0, 0.0, 4.0, 4.0],
        [0.0, 4.0, 0.0, 0.0, 3.0, 4.0, 3.0, 4.0, 0.0, 0.0, 4.0, 0.0, 5.0],
        [5.0, 5.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 4.0, 5.0, 4.0, 0.0, 5.0],
        [4.0, 5.0, 4.0, 5.0, 3.0, 3.0, 0.0, 0.0, 0.0, 3.0, 5.0, 3.0, 0.0],
        [0.0, 3.0, 0.0, 4.0, 3.0, 3.0, 2.0, 0.0, 3.0, 3.0, 4.0, 3.0, 0.0],
        [0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 2.0, 4.0, 3.0, 2.0, 0.0, 3.0],
        [2.0, 4.0, 0.0, 0.0, 3.0, 2.0, 0.0, 4.0, 4.0, 4.0, 0.0, 3.0, 4.0],
        [3.0, 3.0, 0.0, 0.0, 4.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 3.0, 0.0, 2.0, 2.0, 1.0],
        [0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 2.0, 1.0, 0.0],
        [3.0, 2.0, 0.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0, 0.0, 1.0, 0.0, 2.0],
        [0.0, 3.0, 3.0, 1.0, 0.0, 1.0, 0.0, 2.0, 3.0, 2.0, 0.0, 0.0, 1.0],
        ]
        return np.array(ratings)

    def svd(self, A, max_iter=100, tol=1e-10):
        """
        SVD implementation using eigenvalue decomposition
        A = U Σ V^T where:
        - U: left singular vectors from eigenvectors of AA^T
        - Σ: square root of eigenvalues
        - V: right singular vectors from eigenvectors of A^TA
        """
        m, n = A.shape
        k = self.k_factors
        
        # Center the matrix
        mask = A != 0
        row_means = A.sum(axis=1) / mask.sum(axis=1)
        A_centered = A.copy()
        for i in range(m):
            A_centered[i, mask[i]] -= row_means[i]
        
        # A^T A and AA^T
        ATA = np.dot(A_centered.T, A_centered)  # For V
        AAT = np.dot(A_centered, A_centered.T)  # For U
        
        # Eigen Decomposition
        def eigen_decomposition(matrix, k):
            n = matrix.shape[0]
            eigenvectors = np.zeros((n, k))
            eigenvalues = np.zeros(k)
            
            # Initial matrix
            M = matrix.copy()
            
            for i in range(k):
                vector = np.random.randn(n)
                vector = vector / np.linalg.norm(vector)
                
                # Find eigenvector
                for _ in range(max_iter):
                    new_vector = np.dot(M, vector)
                    norm = np.linalg.norm(new_vector)
                    
                    if norm < tol:
                        break
                        
                    new_vector = new_vector / norm
                    
                    # Check convergence
                    if np.abs(np.dot(new_vector, vector)) > 1 - tol:
                        break
                        
                    vector = new_vector
                
                # Calculate eigenvalue
                eigenvalue = np.dot(np.dot(vector.T, M), vector)
                
                # Store results
                eigenvectors[:, i] = vector
                eigenvalues[i] = eigenvalue
                
                # Deflate matrix
                M = M - eigenvalue * np.outer(vector, vector)
            
            return eigenvalues, eigenvectors
        
        # Get eigenvalues and eigenvectors
        eigenvalues_ATA, V = eigen_decomposition(ATA, k)
        eigenvalues_AAT, U = eigen_decomposition(AAT, k)
        
        # Calculate singular values 
        sigma = np.sqrt(np.abs(eigenvalues_ATA))
        
        # Ensure U and V are orthogonal
        U = np.dot(A_centered, V)
        for i in range(k):
            if sigma[i] > tol:
                U[:, i] = U[:, i] / sigma[i]
        
        # Sort singular values and vectors in descending order
        idx = np.argsort(sigma)[::-1]
        sigma = sigma[idx]
        U = U[:, idx]
        V = V[:, idx]
        
        return U, sigma, V.T, row_means

    def predict_ratings(self, U, sigma, Vt, row_means):
        """Predict ratings using SVD"""
        return np.dot(U * sigma, Vt) + row_means.reshape(-1, 1)

    def cosine_similarity(self, matrix):
        """Cosine similarity"""
        norm = np.sqrt(np.sum(matrix ** 2, axis=1))
        matrix_normalized = matrix / norm[:, np.newaxis]
        similarity = np.dot(matrix_normalized, matrix_normalized.T)
        return np.clip(similarity, -1, 1)  # Values are between -1 and 1

    def get_combined_recommendations(self, user_id, ratings_matrix, reconstructed_matrix, 
                                   user_similarity, item_similarity, n_recommendations=5):
        """Get recommendations combining SVD, user-based, and item-based"""
        # Get unrated items
        unrated_items = np.where(ratings_matrix[user_id] == 0)[0]
        
        recommendations = []
        for item in unrated_items:
            # 1. SVD-based rating
            svd_rating = reconstructed_matrix[user_id, item]
            
            # 2. User-based rating
            similar_users = np.argsort(user_similarity[user_id])[::-1][1:4]  # Top 3 similar users
            user_based_ratings = []
            for similar_user in similar_users:
                if ratings_matrix[similar_user][item] != 0:
                    user_based_ratings.append(
                        ratings_matrix[similar_user][item] * user_similarity[user_id][similar_user]
                    )
            user_based_rating = np.mean(user_based_ratings) if user_based_ratings else svd_rating
            
            # 3. Item-based rating
            rated_items = np.where(ratings_matrix[user_id] != 0)[0]
            item_based_ratings = []
            for rated_item in rated_items:
                item_based_ratings.append(
                    ratings_matrix[user_id][rated_item] * item_similarity[item][rated_item]
                )
            item_based_rating = np.mean(item_based_ratings) if item_based_ratings else svd_rating
            
            # Combine ratings (weighted average)
            combined_rating = (0.4 * svd_rating + 
                             0.3 * user_based_rating + 
                             0.3 * item_based_rating)
            
            recommendations.append((item, combined_rating))
        
        # Sort and return top N recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

def main():
    # Initialize system
    rec_sys = RecommendationSystem(k_factors=4)
    
    # Generate sample matrix
    ratings_matrix = rec_sys.create_sample_data()
    print("\nUser-Item Matrix : ")
    print(pd.DataFrame(ratings_matrix).round(2))
    
    # Perform SVD
    U, sigma, Vt, row_means = rec_sys.svd(ratings_matrix)
    print("\nSingular Values:", sigma.round(3))
    print("\n U Matrix:\n",pd.DataFrame(U).round(2))
    print("\n Vt Matrix:\n",pd.DataFrame(Vt).round(2))
    
    # Reconstruct matrix and calculate predictions
    reconstructed_matrix = rec_sys.predict_ratings(U, sigma, Vt, row_means)
    print("\n Predicition Matrix:\n",pd.DataFrame(reconstructed_matrix).round(2))
    
    # Calculate similarities
    user_similarity = rec_sys.cosine_similarity(ratings_matrix)
    item_similarity = rec_sys.cosine_similarity(ratings_matrix.T)
    
    # Plot similarity heatmaps
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(user_similarity, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('User Similarity Matrix')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(item_similarity, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Item Similarity Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Get recommendations for User 1
    user_id = 1
    recommendations = rec_sys.get_combined_recommendations(
        user_id, ratings_matrix, reconstructed_matrix, 
        user_similarity, item_similarity
    )
    
    print(f"\nTop 5 Recommendations for User {user_id}:")
    for item, rating in recommendations:
        print(f"Item {item}: Predicted rating {rating:.2f}")
    
    # Calculate MSE for non-zero entries
    mask = ratings_matrix != 0
    mse = mean_squared_error(ratings_matrix[mask], reconstructed_matrix[mask])
    print(f"\nMSE: {mse:.3f}")

if __name__ == "__main__":
    main()