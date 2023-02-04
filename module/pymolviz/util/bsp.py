import numpy as np

from ..meshes.Mesh import Mesh
from ..meshes.Arrows import Arrows
from ..util.geometries import get_perp


class BSP_Node():

    def __init__(self, polygons = None, plane : tuple = None, _last_normal = None, _last_position = None) -> None:
        if polygons is None:
            self.polygons = []
            self.front = None
            self.back = None
            self.normal = None
            self.position = None
            return
        polygons = list(polygons)
        # split along arbitrary polygon
        if plane is None:
            # heuristic to find a good splitting plane: use arbitrary normal
            # and center of original polygons if there are more than 50
            orig_polys = [poly for poly in polygons if (poly.parent is None)]
            if len(orig_polys) > 50:
                self.position = np.mean([poly.position for poly in orig_polys], axis=0)
                self.normal = np.mean([poly.normal for poly in orig_polys], axis=0)
            else:
                self.position = polygons[0].position
                self.normal = polygons[0].normal
        else:
            self.normal = plane[0]
            self.position = plane[1]
        self.normal = self.normal / np.linalg.norm(self.normal)
        if _last_normal is not None:
            if 1 - np.dot(self.normal, _last_normal) < 1e-5:
                self.normal = get_perp(self.normal); self.normal = self.normal / np.linalg.norm(self.normal)
        if _last_position is not None:
            if np.linalg.norm(self.position - _last_position) < 1e-1:
                self.position = polygons[0].position
                self.normal = polygons[0].normal


        self.polygons = []
        front_polygons = []
        back_polygons = []
        for polygon in polygons:
            front, back, coplanar = polygon.split(self.normal, self.position)
            if front: front_polygons.append(front)
            if back: back_polygons.append(back)
            if coplanar: self.polygons.append(coplanar)

        if len(front_polygons) > 0:
            self.front = BSP_Node(front_polygons, _last_normal = self.normal, _last_position = self.position)
        else:
            self.front = None

        if len(back_polygons) > 0:
            self.back = BSP_Node(back_polygons, _last_normal = self.normal, _last_position = self.position)
        else:
            self.back = None

    def union(self, other):
        c0 = self.copy()
        c1 = other.copy()
        c0.clip_to(c1)
        c1.clip_to(c0)
        return BSP_Node(c0.all_polygons().union(c1.all_polygons()))

    def intersect(self, other):
        c0 = self.copy()
        c1 = other.copy()
        c0.flip()
        c1.flip()
        c0.clip_to(c1)
        c1.clip_to(c0)
        c0.flip()
        c1.flip()
        return BSP_Node(c0.all_polygons().union(c1.all_polygons()))

    def subtract(self, other): 
        c0 = self.copy()
        c1 = other.copy()
        c0.flip()
        c0.clip_to(c1)
        c1.clip_to(c0)
        c0.flip()
        return BSP_Node(c0.all_polygons().union(c1.all_polygons()))


    def plane_mesh(self, scale = 5):
        """ Creates a mesh of the splitting plane.

        Args:
            scale (float, optional): The scale of the plane. Defaults to 5.

        Returns:
            Mesh: A Mesh object.
        """
        from ..meshes.Plane import Plane
        return Plane(self.position, self.normal, scale, "white")
        

    def to_mesh(self, reconstruct = True):
        """ Creates a mesh from the BSP tree.
        
        Returns:
            Mesh: A Mesh object.
        """
        polygons = self.all_polygons(reconstruct=reconstruct)
        return Mesh.combine([poly.to_mesh() for poly in polygons])

    def to_normal_arrows(self):
        """ Creates arrows from the BSP tree.
        
        Returns:
            Arrows: An Arrows object.
        """
        polygons = self.all_polygons()
        return Arrows.combine([poly.normal_mesh() for poly in polygons])
        


    def flip(self):
        """ Flips the BSP tree. Inverting inside and outside.
        
        Returns:
            None
        """
        for polygon in self.polygons:
            polygon.flip()
        self.normal = -self.normal
        if self.front: self.front.flip()
        if self.back: self.back.flip()
        self.front, self.back = self.back, self.front
    
    def from_mesh(mesh: Mesh):
        """ Creates a BSP tree from a mesh.
        
        Args:
            mesh (Mesh): A Mesh object.
        
        Returns:
            BSP_Node: A BSP tree.
        """
        polygons = []
        for face in mesh.faces:
            vertices = mesh.vertices[face]
            # ignore faces where two vertices are the same
            if np.any(np.sum(np.linalg.norm(vertices[None, :, :] - vertices[:, None, :], axis = -1) == 0, axis = 0) > 1):
                continue
            polygons.append(BSP_Polygon(vertices, mesh.color[face], mesh.normals[face]))
        return BSP_Node(polygons)

    def all_polygons(self, reconstruct = False):
        """ Returns all polygons in the BSP tree.
        
        Returns:
            list: A list of BSP_Polygons.
        """

        polygons = set(self.polygons)
        if self.front: 
            front_polygons = self.front.all_polygons()
            polygons.update(front_polygons)
        if self.back: 
            back_polygons = self.back.all_polygons()
            polygons.update(back_polygons)

        if reconstruct:
            new_polygons = set()
            intact = set(polygons)
            while len(polygons) > 0:
                poly = polygons.pop()
                checked, reconstructed = poly.reconstruct(intact)
                polygons.difference_update(checked)
                intact.add(reconstructed)
                new_polygons.add(reconstructed)
            polygons = new_polygons

        return polygons

    def clip_polygons(self, polygons):
        """ Clips a list of polygons against this BSP tree.
        
        Args:
            polygons (list): A list of BSP_Polygons.
        
        Returns:
            list: A list of BSP_Polygons.
        """
        front_polygons = []
        back_polygons = []
        for polygon in polygons:
            front, back, coplanar = polygon.split(self.normal, self.position)
            if front: front_polygons.append(front)
            if back: back_polygons.append(back)
            if coplanar: 
                front_polygons.append(coplanar)
                back_polygons.append(coplanar)


        if self.front:
            front_polygons = self.front.clip_polygons(front_polygons)
        if self.back:
            back_polygons = self.back.clip_polygons(back_polygons)
        else:
            back_polygons = []
        
        return front_polygons + back_polygons

    def clip_to(self, other):
        """ Clips this BSP tree against another BSP tree.
        
        Args:
            other (BSP_Node): A BSP tree.
        
        Returns:
            None
        """
        self.polygons = other.clip_polygons(self.polygons)
        if self.front: self.front.clip_to(other)
        if self.back: self.back.clip_to(other)
    
    def copy(self):
        """ Creates a copy of this BSP tree.
        
        Returns:
            BSP_Node: A BSP tree.
        """
        node = BSP_Node()
        node.normal = self.normal.copy()
        node.position = self.position.copy()
        node.polygons = [poly.copy() for poly in self.polygons]
        if self.front: node.front = self.front.copy()
        if self.back: node.back = self.back.copy()
        return node


class BSP_Polygon():

    def __init__(self, vertices, colors, normals, parent = None) -> None:
        self.vertices = np.array(vertices)
        self.colors = np.array(colors)
        self.normals = np.array(normals)
        self.parent = parent
        self.children = []
        self.position = np.mean(vertices, axis = 0)
        if np.any(np.sum(np.linalg.norm(self.vertices[None, :, :] - self.vertices[:, None, :], axis = -1) == 0, axis = 0) > 1):
            self.normal = np.array([0, 0, 1])
        else:
            self.normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            self.normal /= np.linalg.norm(self.normal)

        if np.any(np.isnan(self.normal)):
            print(f"{len(self.vertices): } had nan, defaulted normal to z axis")
            print(vertices)
            self.normal = np.array([0, 0, 1])

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, BSP_Polygon):
            return np.all(self.vertices == __o.vertices) and np.all(self.colors == __o.colors) and np.all(self.normals == __o.normals)
        return False

    def __hash__(self) -> int:
        return hash((tuple(self.vertices.flatten()), tuple(self.colors.flatten()), tuple(self.normals.flatten())))

    def normal_mesh(self):
        """ Returns a mesh of the normal.
        
        Args:
            None
        
        Returns:
            Arrow: A mesh of the normal.
        """
        return Arrows([self.position, self.position + self.normal], color = "red")

    def copy(self):
        """ Copies the polygon.
        
        Args:
            None
        
        Returns:
            BSP_Polygon: A copy of the polygon.
        """
        return BSP_Polygon(self.vertices, self.colors, self.normals, self.parent)

    def to_mesh(self):
        """ Converts the polygon to a list of faces.
        
        Args:
            None
        
        Returns:
            list: A list of faces.
        """
        faces = []
        for i in range(len(self.vertices) - 2):
            faces.append(np.array([0, i + 1, i + 2]))
        return Mesh(self.vertices, self.colors, self.normals, np.vstack(faces))


    def flip(self):
        """ Flips the polygon.
        
        Args:
            None
        
        Returns:
            None
        """
        self.vertices = self.vertices[::-1]
        self.colors = self.colors[::-1]
        self.normals = self.normals[::-1]
        self.normal *= -1

    def reconstruct(self, intact, checked = None):
        """ Reconstructs this polygons parent if all siblings are intact.
        
        Args:
            intact (list): A list of intact polygons.
            
            Returns:
                Reconstructed siblings, reconstructed polygon.

        """
        if checked is None:
            checked = set()

        if self.parent is None:
            return checked, self


        if all([sibling in intact for sibling in self.parent.children]):
            checked.update(self.parent.children)
            return self.parent.reconstruct(intact, checked)
        else:
            return checked, self


    def split(self, normal, position, tolerance = 1e-9):
        """ Splits the polygon into two polygons based on a plane.
        
        Args:
            normal (np.array): A 3x1 normal vector.
            position (np.array): A 3x1 position vector.
        
        Returns:
            (BSP_Polygon, BSP_Polygon, BSP_Polygon): Front Polygon, Back Polygon, Coplanar Polygon.
        """
        front_vertices = []
        back_vertices = []
        front_color = []
        back_color = []
        front_normals = []
        back_normals = []

        if np.any(np.allclose(self.vertices[None, :, :], self.vertices[:, None, :], atol = tolerance)):
            return None, None, self

        if np.allclose(np.dot(self.vertices - position, normal), 0, atol = 1e-5):
            # all points are coplanar => Polygon is coplanar
            return None, None, self

        for i in range(len(self.vertices)):
            j = (i + 1) % len(self.vertices)
            vi = self.vertices[i]
            vj = self.vertices[j]
            ci = self.colors[i]
            cj = self.colors[j]
            ni = self.normals[i]
            nj = self.normals[j]
            
            # i is front vertex
            if np.dot(normal, vi - position) >= -tolerance:
                front_vertices.append(vi)
                front_color.append(ci)
                front_normals.append(ni)

            # i is back vertex
            if np.dot(normal, vi - position) <= tolerance:
                back_vertices.append(vi)
                back_color.append(ci)
                back_normals.append(ni)
            

            # i and j are on different sides of the plane (spanning):
            # we need to insert a middle point
            if (np.dot(normal, vi - position) * np.dot(normal, vj - position) < 0) \
                 and (not np.abs(np.dot(normal, vj - vi)) < tolerance): 
                t = np.dot(normal, position - vi) / np.dot(normal, vj - vi)
                if (t > tolerance) and (t < 1 - tolerance):
                    v = vi + t * (vj - vi)
                    c = ci + t * (cj - ci)
                    n = ni + t * (nj - ni)
                    front_vertices.append(v)
                    front_color.append(c)
                    front_normals.append(n)
                    back_vertices.append(v)
                    back_color.append(c)
                    back_normals.append(n)
            
        front_vertices = np.array(front_vertices)
        back_vertices = np.array(back_vertices)
        
        if (len(front_vertices) >= 3) and (len(back_vertices) >= 3):
            front = BSP_Polygon(front_vertices, front_color, front_normals, parent=self)
            back = BSP_Polygon(back_vertices, back_color, back_normals, parent=self)
            self.children = [front, back]
            return front, back, None

        elif len(front_vertices) >= 3:
            return self, None, None

        elif len(back_vertices) >= 3:
            return None, self, None


