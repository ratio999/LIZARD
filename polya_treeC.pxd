# cython: language_level=3
cimport numpy as cnp
cimport function_treeC


cdef class PolyaNode:
    cdef public str kind
    cdef public int n_choices
    cdef public int depth
    cdef public cnp.ndarray alphas
    cdef public double alpha_total
    cdef public dict children

    cpdef double log_prob(self, int choice_idx)
    cpdef int sample_choice(self)
    cpdef void update(self, int choice_idx, double gamma)


cdef class PolyaTree:
    cdef public PolyaNode root
    cdef public list operations
    cdef public list primitives
    cdef public double alpha_0
    cdef public int t
    cdef public int n_ops
    cdef public int n_prims
    cdef public dict _op_to_idx
    cdef public dict _prim_to_idx

    cdef PolyaNode _ensure_child(self, PolyaNode parent, str kind, int choice_idx)
    cdef PolyaNode _get_child_or_none(self, PolyaNode parent, str kind, int choice_idx)
    cdef int _root_index(self, str kind, int choice_idx)

    cpdef double log_prior(self, list construction_path)
    cpdef void update_path(self, list construction_path)
    cpdef list extract_path(self, function_treeC.Node node)
    cdef void _extract_recursive(self, function_treeC.Node node, list path)
    cpdef double log_prior_from_tree(self, function_treeC.Node root_node)
    cpdef list sample_construction_path(self, int max_steps=?)
    cpdef int add_primitive(self, str desc)
    cpdef add_primitives(self, list descs)
