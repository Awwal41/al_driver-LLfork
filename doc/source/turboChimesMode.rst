.. _page-turboChimes:

***************************************
Multilayer (TurboChIMES) Fitting Mode
***************************************

.. figure:: turbo_ChIMES.png
   :width: 500
   :align: center

   **Fig. 1.** Schematic of the TurboChIMES / multilayer idea: short-ranged interactions are represented with a **dense** Chebyshev basis (bond rearrangements, steep repulsion), while long-ranged contributions use a **sparser** basis that varies more smoothly.

This page documents the **multilayer ChIMES** workflow as implemented in the Active Learning Driver (ALD). It is complementary to :ref:`page-basic` (single-layer iterative refinement), :ref:`page-hierarch` (element-wise parameter hierarchy and transfer), and :ref:`page-clusterAL` (cluster-based active learning). If you are deciding which mode to use: **hierarchical** fitting addresses complexity across **chemical species**; **multilayer (TurboChIMES)** addresses how much **resolution** to spend at short vs. long range for the *same* species and training set for simulation efficiency.

.. figure:: short_and_long_range_description.png
   :width: 650
   :align: center

   **Fig. 2.** Short- vs. long-range resolution in the multilayer construction (rasterized from the manuscript figure; vector PDF also available).

A vector version is still provided for slides or print: :download:`short_and_long_range_description.pdf <short_and_long_range_description.pdf>`.

-------

====================================
Motivation and mathematical overview
====================================

Machine-learned interatomic potentials (ML-IAPs) can achieve near–quantum accuracy for covalently bonded systems, but their **computational cost** scales with the richness of the basis used everywhere in space. A natural design choice is how to distribute that resolution across **distance**: short range is where bond rearrangement and strong repulsion dominate, while longer-range forces vary more smoothly.

The **multilayer ChIMES** representation uses a **dense** basis for short-ranged interactions and a **sparser** basis for longer-ranged contributions, implemented within the ChIMES ML-IAP framework. In studies on reactive and non-reactive molecular fluids over broad thermodynamic conditions, this strategy **preserves accuracy** while **reducing evaluation cost** relative to a conventional single-layer construction—including order-of-magnitude speedups for representative systems.

**Two-layer construction (short + long).** The user specifies ChIMES hyperparameters **independently** for each layer: in particular, maximum **bodiedness**, maximum **polynomial orders** (:math:`\mathcal{O}_{n\mathrm{B}}`), and **outer cutoffs** :math:`r_{\mathrm{cut}}`. ChIMES LSQ then builds a design matrix **per layer**, :math:`\mathbf{M}_{\mathrm{short}}` and :math:`\mathbf{M}_{\mathrm{long}}`, with the **same** number of rows (one per constraint row derived from the reference data :math:`\mathbf{X}_{\mathrm{DFT}}`) but **different** numbers of columns when the two layers use different bodiedness / polynomial orders. The matrices are **concatenated horizontally** and the combined linear system is solved in weighted least-squares form:

.. math::
   :label: eq-turbo-design

   \mathbf{w}
   \left[\mathbf{M}_{\mathrm{short}}\ \mathbf{M}_{\mathrm{long}}\right]
   \begin{bmatrix}
   \mathbf{c}_{\mathrm{short}} \\ \mathbf{c}_{\mathrm{long}}
   \end{bmatrix}
   =
   \mathbf{w}\,\mathbf{X}_{\mathrm{DFT}}

Here :math:`\mathbf{w}` denotes the usual weighting of rows (forces, energies, stresses, etc., as in standard ChIMES LSQ). The coefficient blocks :math:`\mathbf{c}_{\mathrm{short}}` and :math:`\mathbf{c}_{\mathrm{long}}` are the **separate** parameters for the short- and long-ranged layers. At **run time**, LAMMPS evaluates the two ChIMES pair contributions and combines them (e.g. via ``hybrid/overlay`` style pair interactions), as produced by the ALD/ChIMES tool chain.

For the relationship between ``N_HYPER_SETS``, ``fm_setup`` files, and driver behavior, see also :ref:`page-options` (ChIMES LSQ and **Multilayer** subsection).

-------

============================
Example fit: Propane system
============================

.. Note::

   Example files live under:

   ``<al_driver repository>/examples/simple_iter_single_statepoint-lmp-test-turbo-test``

This example illustrates an iterative fit for **united-atom propane** (:math:`\texttt{C1}`, :math:`\texttt{C2}` labels in the force field). It highlights **intra-molecular** (short-ranged) structure within each propane unit and **inter-molecular** (longer-ranged) van der Waals interactions between molecules in the box. Reference data for this bundled example use **LAMMPS** as the “QM” reference method (``BULK_QM_METHOD = "LMP"``), so you must supply both **ChIMES MD** templates and **classical reference** LAMMPS inputs (see below).

------------------------------------------
Directory layout (relative to the example)
------------------------------------------

Compared with :ref:`page-basic`, the ``ALC-0_BASEFILES`` area contains **two** numbered setup files instead of a single ``fm_setup.in``. The ``LMPMD_BASEFILES`` and ``LMP_BASEFILES`` directories supply ChIMES fitting MD and reference LAMMPS jobs, respectively.

.. code-block:: text
   :emphasize-lines: 4,14-16

   ALL_BASE_FILES/
   ├── ALC-0_BASEFILES/
   │   ├── 0.fm_setup.in          # short-range hyperparameters / cutoffs
   │   ├── 1.fm_setup.in          # long-range hyperparameters / cutoffs
   │   ├── test_data.xyzf
   │   └── traj_list.dat
   ├── LMP_BASEFILES/
   │   ├── 1.data.in
   │   └── 1.in.lammps
   └── LMPMD_BASEFILES/
       ├── bonds.dat
       ├── case-0.indep-0.data.in
       ├── case-0.indep-0.in.lammps
       └── case-0.skip.dat

**Role of ``0.fm_setup.in`` vs. ``1.fm_setup.in``.** Layer ``0`` encodes the **short-ranged** ChIMES model (typically smaller outer cutoffs and/or higher polynomial rank where the potential is stiff). Layer ``1`` encodes the **long-ranged** model (larger cutoffs and a sparser Chebyshev expansion where the potential is smoother). Your exact hyperparameter lines must be self-consistent with the ChIMES LSQ manual and the physical system.

-------

------------------------------------------
Input files and ``config.py``
------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Key ALD settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multilayer driver path is selected when ``N_HYPER_SETS`` is set to **2** (in general, to the number of independent ``*.fm_setup.in`` layers you intend to fit). With ``N_HYPER_SETS = 1``, the ALD follows :ref:`page-basic` or :ref:`page-hierarch` depending on ``DO_HIERARCH`` and related options—not this page.

**LAMMPS is required for MD** in the Turbo workflow as shipped: the fitted potential is expressed as **two** ChIMES pair styles that LAMMPS combines (e.g. ``hybrid/overlay``). Other ``MD_STYLE`` choices are not described here.

A minimal excerpt from the example ``config.py`` (compare full file in the example directory):

.. code-block:: python
   :linenos:
   :emphasize-lines: 26-27

   ################################
   ##### General variables
   ################################

   ATOM_TYPES = ['C']
   NO_CASES = 1

   DRIVER_DIR     = "/path/to/al_driver/src/"
   WORKING_DIR    = "/path/to/examples/simple_iter_single_statepoint-lmp-test-turbo-test/"
   CHIMES_SRCDIR  = "/path/to/chimes_lsq/src/"

   ################################
   ##### ChIMES LSQ
   ################################

   ALC0_FILES    = WORKING_DIR + "ALL_BASE_FILES/ALC-0_BASEFILES/"
   CHIMES_LSQ    = CHIMES_SRCDIR + "../build/chimes_lsq"
   CHIMES_SOLVER = CHIMES_SRCDIR + "../build/chimes_lsq.py"
   CHIMES_POSTPRC= CHIMES_SRCDIR + "../build/post_proc_chimes_lsq.py"

   WEIGHTS_FORCE = 1.0
   REGRESS_ALG   = "dlasso"
   REGRESS_VAR   = "1.0E-5"
   REGRESS_NRM   = True
   N_HYPER_SETS  = 2

   ################################
   ##### Molecular Dynamics
   ################################

   MD_STYLE        = "LMP"
   MDFILES         = WORKING_DIR + "/ALL_BASE_FILES/LMPMD_BASEFILES/"
   MD_MPI          = "/path/to/lmp_mpi_chimes"
   CHIMES_MD_MPI   = MD_MPI
   MOLANAL         = CHIMES_SRCDIR + "../contrib/molanal/src/"
   MOLANAL_SPECIES = ["C1"]

   ################################
   ##### Single-point "QM" (LAMMPS reference)
   ################################

   BULK_QM_METHOD  = "LMP"
   IGAS_QM_METHOD  = "LMP"
   QM_FILES        = WORKING_DIR + "ALL_BASE_FILES/LMP_BASEFILES"
   LMP_EXE         = "/path/to/lmp_mpi_chimes"   # must support class2 / molecular reference inputs
   LMP_UNITS       = "REAL"

Replace all paths, queues, modules, and account settings with those appropriate for your HPC environment. The reference LAMMPS job in ``LMP_BASEFILES`` uses **molecular** atom styles and **class2** angles in the bundled example; your ``LMP_EXE`` must be built with compatible packages (see :ref:`page-basic` LAMMPS discussion where applicable).

-------

------------------------------------------
Running and inspecting output
------------------------------------------

The execution pattern matches :ref:`page-basic`:

1. Edit ``config.py`` and all paths under ``ALL_BASE_FILES/``.
2. From ``WORKING_DIR``, run the driver, e.g. ``python3 /path/to/al_driver/src/main.py 0 1 2 3`` (or your site’s wrapper), preferably inside ``screen`` or ``tmux``.

Inspect ChIMES LSQ logs, fitted parameter files for **both** layers, and subsequent LAMMPS trajectories as you would for a single-layer fit. If regression fails or weights are imbalanced, revisit ``WEIGHTS_*``, ``REGRESS_VAR``, and the hyperparameters in ``0.fm_setup.in`` / ``1.fm_setup.in``.

-------

========================================================
Further reading and cross-links
========================================================

* **Setup and option tables:** :ref:`page-options` (including ``N_HYPER_SETS`` and the **Multilayer (TurboChIMES)** subsection immediately after hierarchical options).
* **Single-layer baseline:** :ref:`page-basic`.
* **Species-wise transfer (different problem):** :ref:`page-hierarch`.
* **Cluster-based AL:** :ref:`page-clusterAL`.
