import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

# Keep the previous AdaptiveBivariatePolyaTreeMultiAlpha class
class AdaptiveBivariatePolyaTreeMultiAlpha:
    def __init__(self, base_depth=4, max_depth=8, density_threshold=0.01, min_points=5, 
                 alpha0=0.5, lam=1.0, beta=0.5, gamma=2.0, eta=1.0, domain=((0, 1), (0, 1))):
        self.base_depth = base_depth
        self.max_depth = max_depth
        self.density_threshold = density_threshold
        self.min_points = min_points
        self.alpha0 = alpha0
        self.lam = lam
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.domain = domain
        self.counts = {}
        self.counts[()] = 0
        self.total_points = 0
        
    # Previous methods remain the same
    def _get_quadrant(self, x, y, x_mid, y_mid):
        if x <= x_mid:
            return 0 if y <= y_mid else 1
        else:
            return 2 if y <= y_mid else 3
    
    def _get_node_bounds(self, node):
        x_left, x_right = self.domain[0]
        y_bottom, y_top = self.domain[1]
        for q in node:
            x_mid = (x_left + x_right) / 2.0
            y_mid = (y_bottom + y_top) / 2.0
            if q in [0, 1]:
                x_right = x_mid
            else:
                x_left = x_mid
            if q in [0, 2]:
                y_top = y_mid
            else:
                y_bottom = y_mid
        return (x_left, x_right), (y_bottom, y_top)
    
    def _compute_local_density(self, node):
        (x_left, x_right), (y_bottom, y_top) = self._get_node_bounds(node)
        area = (x_right - x_left) * (y_top - y_bottom)
        count = self.counts.get(node, 0)
        return count / (self.total_points * area) if area > 0 else 0
    
    def _should_split(self, node, count):
        if len(node) >= self.max_depth:
            return False
        if len(node) < self.base_depth:
            return True
        density = self._compute_local_density(node)
        return (count >= self.min_points and density >= self.density_threshold and len(node) < self.max_depth)
    
    def _update_counts(self, points):
        self.total_points = len(points)
        points_to_process = [((), points)]
        while points_to_process:
            node, node_points = points_to_process.pop(0)
            if len(node_points) == 0:
                continue
            self.counts[node] = len(node_points)
            if self._should_split(node, len(node_points)):
                (x_left, x_right), (y_bottom, y_top) = self._get_node_bounds(node)
                x_mid = (x_left + x_right) / 2.0
                y_mid = (y_bottom + y_top) / 2.0
                quadrants = {0: [], 1: [], 2: [], 3: []}
                for p in node_points:
                    x, y = p
                    q = self._get_quadrant(x, y, x_mid, y_mid)
                    quadrants[q].append(p)
                for q in range(4):
                    new_node = node + (q,)
                    points_to_process.append((new_node, quadrants[q]))
    
    def fit(self, points):
        self._update_counts(points)
        return self
    
    # Modified method to also return pixel areas
    def compute_alphas_density_and_areas(self, X, Y):
        densities = np.zeros_like(X)
        pixel_areas = np.zeros_like(X)
        alpha_maps = {
            'depth': np.zeros_like(X),
            'density': np.zeros_like(X),
            'curvature': np.zeros_like(X),
            'hybrid': np.zeros_like(X)
        }
        normalized_alpha_maps = {
            'depth': np.zeros_like(X),
            'density': np.zeros_like(X),
            'curvature': np.zeros_like(X),
            'hybrid': np.zeros_like(X)
        }
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i,j], Y[i,j]
                node = ()
                prob = 1.0
                while True:
                    (x_left, x_right), (y_bottom, y_top) = self._get_node_bounds(node)
                    area = (x_right - x_left) * (y_top - y_bottom)
                    pixel_areas[i,j] = area
                    
                    local_density = self._compute_local_density(node)
                    depth = len(node)
                    
                    # Different alpha definitions
                    alpha_depth = self.alpha0 * (1 + self.lam * depth / self.max_depth)
                    alpha_density = self.alpha0 * (1 + self.beta * np.exp(-self.gamma * local_density))
                    alpha_curvature = self.alpha0 * (1 + self.eta * abs(depth * local_density))
                    alpha_hybrid = alpha_depth * (1 + self.beta * np.exp(-self.gamma * local_density))
                    
                    # Store regular alpha values
                    alpha_maps['depth'][i,j] = alpha_depth
                    alpha_maps['density'][i,j] = alpha_density
                    alpha_maps['curvature'][i,j] = alpha_curvature
                    alpha_maps['hybrid'][i,j] = alpha_hybrid
                    
                    # Store normalized alpha values (alpha/area)
                    normalized_alpha_maps['depth'][i,j] = alpha_depth / area if area > 0 else 0
                    normalized_alpha_maps['density'][i,j] = alpha_density / area if area > 0 else 0
                    normalized_alpha_maps['curvature'][i,j] = alpha_curvature / area if area > 0 else 0
                    normalized_alpha_maps['hybrid'][i,j] = alpha_hybrid / area if area > 0 else 0
                    
                    if not self._should_split(node, self.counts.get(node, 0)):
                        densities[i,j] = prob * local_density
                        break
                        
                    x_mid = (x_left + x_right) / 2
                    y_mid = (y_bottom + y_top) / 2
                    quadrant = self._get_quadrant(x, y, x_mid, y_mid)
                    node = node + (quadrant,)
                    prob /= 4.0
                    
        return densities, alpha_maps, normalized_alpha_maps, pixel_areas

# Generate synthetic chirp data
def generate_chirp_data(n_points=2000):
    t = np.linspace(0, 1, n_points)
    f = 20 + 980 * t**2
    t = t + 0.02 * np.random.randn(n_points)
    f = f + 20 * np.random.randn(n_points)
    t = (t - t.min()) / (t.max() - t.min())
    f = (f - f.min()) / (f.max() - f.min())
    return np.column_stack([t, f])

# Generate and fit data
np.random.seed(42)
points = generate_chirp_data()
t_points = points[:, 0]
f_points = points[:, 1]

# Setup grid for density estimation
resolution = 100
x = np.linspace(0, 1, resolution)
y = np.linspace(0, 1, resolution)
X, Y = np.meshgrid(x, y)
positions = np.vstack([X.ravel(), Y.ravel()])

# Kernel Density Estimation
kernel = gaussian_kde(points.T)
Z = np.reshape(kernel(positions), X.shape)

print(Z.shape)
# Plot as a time-frequency map
plt.figure(figsize=(12, 8))
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Time-Frequency Map Representation of Chirp Data')
plt.show()

print('Chirp data represented as time-frequency map.')
aptree = AdaptiveBivariatePolyaTreeMultiAlpha(
    base_depth=4, max_depth=8,
    density_threshold=0.01, min_points=5,
    alpha0=0.5, lam=1.0, beta=0.5, gamma=2.0, eta=1.0
)
aptree.fit(points)

# Create evaluation grid
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# Compute densities, alphas, and areas
densities, alpha_maps, normalized_alpha_maps, pixel_areas = aptree.compute_alphas_density_and_areas(X, Y)

# Create two figures: one for regular alphas (as before) and one for normalized alphas
fig1 = plt.figure(figsize=(20, 15))
fig1.suptitle('Regular Alpha Maps', fontsize=16)

# Original data
ax1 = fig1.add_subplot(321)
ax1.scatter(points[:,0], points[:,1], alpha=0.1, s=1)
ax1.set_title('Original Chirp Data')
ax1.set_xlabel('Time')
ax1.set_ylabel('Frequency')

# Plot for each alpha definition
titles = ['Depth-based Alpha', 'Density-based Alpha', 
          'Curvature-based Alpha', 'Hybrid Alpha']
positions = [322, 323, 324, 325]

for title, pos, (key, alpha_map) in zip(titles, positions, alpha_maps.items()):
    ax = fig1.add_subplot(pos)
    im = ax.pcolormesh(X, Y, alpha_map, shading='auto', cmap='plasma')
    ax.set_title(f'{title} (α map)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax)

# Density estimate
ax6 = fig1.add_subplot(326)
im = ax6.pcolormesh(X, Y, densities, shading='auto', cmap='viridis')
ax6.set_title('Estimated Density')
ax6.set_xlabel('Time')
ax6.set_ylabel('Frequency')
plt.colorbar(im, ax=ax6)

plt.tight_layout()
plt.show()

# New figure for normalized alphas
fig2 = plt.figure(figsize=(20, 15))
fig2.suptitle('Normalized Alpha Maps (α/pixel_area)', fontsize=16)

# Pixel areas
ax1 = fig2.add_subplot(321)
im = ax1.pcolormesh(X, Y, pixel_areas, shading='auto', cmap='viridis')
ax1.set_title('Pixel Areas')
ax1.set_xlabel('Time')
ax1.set_ylabel('Frequency')
plt.colorbar(im, ax=ax1)

# Plot for each normalized alpha definition
for title, pos, (key, norm_alpha_map) in zip(titles, positions, normalized_alpha_maps.items()):
    ax = fig2.add_subplot(pos)
    im = ax.pcolormesh(X, Y, norm_alpha_map, shading='auto', cmap='plasma')
    ax.set_title(f'{title} (α/area)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax)

# Log of normalized hybrid alpha (to better see the variations)
ax6 = fig2.add_subplot(326)
log_norm_hybrid = np.log(normalized_alpha_maps['hybrid'] + 1e-10)
im = ax6.pcolormesh(X, Y, log_norm_hybrid, shading='auto', cmap='viridis')
ax6.set_title('Log of Normalized Hybrid Alpha')
ax6.set_xlabel('Time')
ax6.set_ylabel('Frequency')
plt.colorbar(im, ax=ax6)

plt.tight_layout()
plt.show()