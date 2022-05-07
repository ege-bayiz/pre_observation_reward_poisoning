import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def gen_theta_corr_noise(A, theta, K, T):
    noise = np.empty((K,T))
    noise[:, 0] = A * gen_rand_unit_vec(K)

    for ii in range(1,T):
        r = np.random.multivariate_normal(np.zeros_like(noise[:, ii - 1]), np.eye(K))
        perp = r - r.dot(noise[:, ii - 1]) * noise[:, ii - 1]  # Find perp. component
        noise[:, ii] = np.cos(theta) * noise[:, ii - 1] + np.sqrt(1 - np.cos(theta) ** 2) * perp  # find component with angle theta
        noise[:, ii] = noise[:, ii] / np.linalg.norm(noise[:, ii])  # Normalize

    return noise


def gen_rand_unit_vec(dim):
    vec = np.random.multivariate_normal(np.zeros(dim), 100 * np.eye(dim))  # Generate Gaussian noise
    return vec / np.linalg.norm(vec)  # Normalize


# Plotting
noise = gen_theta_corr_noise(1, 0.5, 10, 100)
df = pd.DataFrame(noise.transpose())
print(df.head())

fig = plt.figure(figsize=(4,6))
sns.lineplot(data=df)
plt.show()