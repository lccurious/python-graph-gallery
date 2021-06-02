import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs


def mean_by_label(samples, labels, num_classes):
    """Select mean(sample), count() from samples group by labels order by labels asc

    :param samples: NxM matrix where N is number of samples and M is number of dimension
    :type samples: np.ndarray
    :param labels: Nx1 labels array
    :type labels: np.ndarray
    :param num_classes: The complete number of categories
    :type num_classes: int
    """
    weight = np.zeros((num_classes, samples.shape[0]), dtype=samples.dtype)
    weight[labels, np.arange(samples.shape[0])] = 1
    weight = weight / np.linalg.norm(weight, axis=1, ord=1, keepdims=True)
    mean = np.dot(weight, samples)
    return mean


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class ClassificationBall(object):
    def __init__(self) -> None:
        super().__init__()
        self.x = 0
        self.y = 0
        self.z = 0
        self.center = (0.0, 0.0, 0.0)
        self.data = None
        self.labels = None
        self.local_anchors = None
    
    def create_ball(self, center=(0.0, 0.0, 0.0)):
        """Initialize a 3D ball
        
        :param center: The center coordinate of ball
        :type center: tuple
        """
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        self.x = 1. * np.outer(np.cos(u), np.sin(v))
        self.y = 1. * np.outer(np.sin(u), np.sin(v))
        self.z = 1. * np.outer(np.ones(np.size(u)), np.cos(v))

        self.x += center[0]
        self.y += center[1]
        self.z += center[2]
        self.center = center
    
    def show(self, ax=None, color='b', plot_sample=True, plot_ball=True, plot_quiver=True, plot_displacement=False, label='source'):
        """Show the dataset in ball

        :param ax: axis, if is None, create a new figure else use the ax
        :type ax: plt.Axis, optional
        """
        cx, cy, cz = self.center
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        if plot_ball:
            ax.plot_wireframe(self.x, self.y, self.z, 
                              rstride=4, cstride=4, color=color, linewidth=1.0, alpha=0.1, label=f'{label} envelope')

        X, y = self.data, self.labels
        ax.scatter(cx, cy, cz, marker='*', color=color, label=f'{label} barycenter')
        if plot_sample:
            ax.scatter(X[:, 0]+cx, X[:, 1]+cy, X[:, 2]+cz, color=color, label=f'{label} samples')
        
        local_anchors = mean_by_label(X, y, len(np.unique(y)))
        ax.scatter(local_anchors[:, 0] + cx, local_anchors[:, 1] + cy, local_anchors[:, 2] + cz, 
                   color=color, marker='s', label=f'{label} local anchor')
        ax.quiver(cx, cy, cz, local_anchors[:, 0], local_anchors[:, 1], local_anchors[:, 2],
                  length=1.0, arrow_length_ratio=0.1,
                  normalize=True, color=color, alpha=0.8, label=f'{label} discriminative vector')
        
        if plot_quiver:
            ax.quiver(cx, cy, cz, X[:, 0], X[:, 1], X[:, 2],
                      length=1.0, arrow_length_ratio=0.1,
                      normalize=True, color=color, alpha=0.3, label=f'{label} vectors')
            
    def generate_points(self, centers=None, num_samples=100, cluster_std=0.1):
        """Generate some random points around the ball surface.

        :param centers: given centers for blobs generation
        :type centers: np.array
        :param num_samples: number of samples in total, defaults to 30
        :type num_samples: int, optional
        """
        X, y = make_blobs(n_samples=num_samples, centers=centers, n_features=3, 
                          cluster_std=cluster_std) 
        # normalize X
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.data = X
        self.labels = y
        self.local_anchors = mean_by_label(self.data, self.labels, len(np.unique(self.labels)))
    
    def load_points(self, filename):
        """Load file from save data

        :param filename: Filename
        :type filename: str
        """
        data = np.load(filename, allow_pickle=True).item()
        self.data = data['X'] * (0.7 + np.random.rand(data['X'].shape[0], 1) * 0.3)
        self.labels = data['y']
        self.local_anchors = mean_by_label(self.data, self.labels, len(np.unique(self.labels)))


if __name__ == '__main__':
    ball_src = ClassificationBall()
    ball_tgt = ClassificationBall()

    centers = np.random.rand(8, 3) * 2 - 1.0

    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Generate Onece
    # ball_src.generate_points(centers)
    # np.save('source_points.npy', {'X': ball_src.data, 'y': ball_src.labels})
    # ball_tgt.generate_points(centers)
    # np.save('target_points.npy', {'X': ball_tgt.data, 'y': ball_tgt.labels})

    # Load data
    ball_src.load_points('source_points.npy')
    ball_tgt.load_points('target_points.npy')

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ball_src.create_ball()
    ball_src.show(ax, color='r', plot_ball=True, plot_quiver=False, label='source')
    ball_tgt.create_ball(center=(0.25, 0.0, 0.2))
    ball_tgt.show(ax, plot_ball=True, plot_quiver=False, label='target')

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    set_axes_equal(ax)
    xyz_lim = [-1.0, 1.25]
    # ax.set_xlim3d(xyz_lim)
    # ax.set_ylim3d(xyz_lim)
    # ax.set_zlim3d(xyz_lim)
    ax.axis('off')
    # ax.legend()
    plt.tight_layout()

    plt.show()
    fig.savefig('ball_points_visualizatoin.pdf', transparent=True, dpi=300)
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ball_src.create_ball()
    ball_src.show(ax, color='r', plot_ball=True, plot_sample=False, plot_quiver=False, label='source')
    ball_tgt.create_ball(center=(0.25, 0.0, 0.2))
    ball_tgt.show(ax, plot_ball=True, plot_sample=False, plot_quiver=False, label='target')

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    set_axes_equal(ax)
    xyz_lim = [-1.0, 1.25]
    # ax.set_xlim3d(xyz_lim)
    # ax.set_ylim3d(xyz_lim)
    # ax.set_zlim3d([l * 0.9 for l in xyz_lim])
    ax.axis('off')
    ax.legend()
    plt.tight_layout()

    plt.show()
    fig.savefig('ball_visualizatoin.pdf', transparent=True, dpi=300)
