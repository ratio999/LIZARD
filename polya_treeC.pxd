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
    # Phase 2.10 (DP completion): base mass of the UNREGISTERED remainder
    # cell R at this node. Never reinforced, never proposed, EXCLUDED from
    # the predictive normaliser (restriction by neutrality, THEORY §5.4).
    # 0.0 in legacy mode -> bit-identical legacy behaviour.
    cdef public double alpha_rem

    cdef double log_prob(self, int choice_idx)
    cdef int sample_choice(self)
    cdef void update(self, int choice_idx, double gamma)


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
    cdef public bint auto_register   # auto add_primitive on unknown leaves
    cdef public bint strict_ops      # KeyError on unknown ops (default True)
    # Phase 2.10 (DP completion) — see THEORY_2_13.md §5 / DESIGN_2_10:
    # dp_mode        : True iff constructed with remainder_mass > 0.
    # rem_base       : current unallocated remainder base mass (same at
    #                  every node; carving keeps all replicas in sync).
    # base_alphas    : per registered flat cell, the BASE mass c*G0(cell)
    #                  used to initialise fresh (lazily created) nodes.
    # base_reg_total : sum(base_alphas) — normaliser for the unvisited-
    #                  region predictive in dp_mode.
    cdef public bint dp_mode
    cdef public double rem_base
    cdef public list base_alphas
    cdef public double base_reg_total
    # Phase 2.10 Stage 3c (vocabulary freeze): unify_params=True maps every
    # PLAIN parameter leaf to the single canonical psi_const cell
    # (PARAM_CELL_TOKEN, appended to `primitives` at construction) instead
    # of one cell per parameter name — names are bound labels whose indices
    # new_par can ratchet without bound, so per-name cells can neither be
    # frozen nor assign alpha-equivalent trees equal mass. _param_idx is
    # psi_const's flat primitive index (-1 when not unified).
    cdef public bint unify_params
    cdef public int _param_idx
    # Phase 4.3 (frozen-regime ablation, 2026-07-16): opt-in switch that
    # halts urn reinforcement (update_path becomes a no-op) so Theta is
    # FROZEN at its current value — the paper's "frozen regime". Default
    # False = continuously adaptive = bit-identical legacy behaviour.
    cdef public bint adapt_frozen

    cdef PolyaNode _ensure_child(self, PolyaNode parent, str kind, int choice_idx)
    cdef PolyaNode _get_child_or_none(self, PolyaNode parent, str kind, int choice_idx)
    cdef int _root_index(self, str kind, int choice_idx)

    cpdef double log_prior(self, list construction_path)
    cdef void update_path(self, list construction_path)
    cpdef list extract_path(self, function_treeC.Node node)
    cdef void _legacy_extract_recursive(self, function_treeC.Node node, list path)
    cdef void _extract_recursive(self, function_treeC.Node node, list path)
    cpdef double log_prior_from_tree(self, function_treeC.Node root_node)
    cpdef list sample_construction_path(self, int max_steps=?)
    cpdef int add_primitive(self, str desc, object mass=?)
    cpdef add_primitives(self, list descs)
    cdef void _grow_all_nodes(self, double new_mass)

    cpdef assert_covers(self, function_treeC.Node root_node)
    cdef _assert_covers_recursive(self, function_treeC.Node node)

    # Phase 2.12 (MPI sampling): picklable snapshot / restore / replay.
    cpdef dict get_state(self)
    cpdef void set_state(self, dict state) except *
    cpdef void replay_paths(self, list paths) except *

cdef dict _node_to_state(PolyaNode node)
cdef PolyaNode _node_from_state(dict s)