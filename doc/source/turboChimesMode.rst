.. _page-turboChimes:

***************************************
Multilayer (TurboChIMES) Fitting Mode
***************************************

.. figure:: short_and_long_range_description.png
   :width: 325
   :align: center

   Fig. 1. Short- vs. long-range resolution in the multilayer construction.

A .pdf version of this figure is available: :download:`short_and_long_range_description.pdf <short_and_long_range_description.pdf>`.

This page documents the multilayer ChIMES (TurboChIMES) workflow as implemented in the Active Learning Driver (ALD).

-------

====================================
Motivation and mathematical overview
====================================

Covalently bonded fluids and molecular materials combine stiff short-ranged bonded interactions, many-body correlations in the first coordination shell, and smoother longer-ranged non-bonded forces. Machine-learned interatomic potentials (ML-IAPs) are often trained to represent all of these contributions in a single unified model. Consequently, necessary basis set completeness has been governed by featuredness of short-range/bonded interactions, while interaction range has been governed by relatively smooth mid-to-long-range/non-bonded interactions; see Fig. 1. However, these systems could be more efficiently described by simultaneously learning two overlaid ML-IAPs: one that is short ranged and described through many basis functions, and another that is relatively longer ranged and requires fewer. The result would be models that exhibit the same accuracy as those developed through the single-layer strategy, while exhibiting greater computational efficiency due to the lower overall number of basis functions needed. This page details how this strategy is realized within the context of the ChIMES ML-IAP workflow in the Active Learning Driver. For the method and benchmarks, see the multilayer ChIMES manuscript `(link forthcoming) <https://arxiv.org/abs/TBD>`_.

Multilayer ChIMES requires tuning maximum polynomial orders for both the short- and long-range layers, as well as specification of the short-range outer cutoff. Based on this study, we establish the following heuristic guidelines:

1. :math:`r_{\mathrm{cut,short}}` can be selected based on inspection of the ground-truth radial distribution function (RDF).
2. Optimal polynomial orders can be substantially lower than what is typically used for single-layer models; in representative fits, average short-range orders were about half those used for a comparable single-layer ChIMES model.
3. Optimal long-range orders are generally lower than those for the short-range layer; in many tests, a practical long-range starting point was :math:`\mathcal{O}_{\mathrm{2B,long}} = 4` with higher-body long-range orders set to zero.
4. :math:`\mathcal{O}_{\mathrm{2B,short}}` and :math:`\mathcal{O}_{\mathrm{3B,short}}` can be optimized independently of :math:`r_{\mathrm{cut,short}}`, :math:`\mathcal{O}_{\mathrm{2B,long}}`, and :math:`\mathcal{O}_{\mathrm{3B,long}}`.
5. :math:`r_{\mathrm{cut,short}}` can be optimized independently of any polynomial order.

For the multilayer water model, different :math:`r_{\mathrm{cut,short}}` values were used for each atom-pair type. The values were taken as the location of the first minimum in the ground-truth 300 K RDF after the last bonded peak. In ``fm_setup.in``, set these as pair-specific ``S_MAXIM`` values in the short-range layer file (layer ``0`` in the bundled examples). The long-range layer then extends to the full non-bonded outer cutoff on the corresponding ``PAIRIDX`` rows.

Hyperparameters in each numbered ``fm_setup.in`` should follow the guidance provided in the `ChIMES LSQ manual <https://chimes-lsq.readthedocs.io/en/latest/index.html>`_ and be reasonable for describing your target physical system. As a starting point for a two-layer fit, use higher 2- and 3-body orders with shorter ``S_MAXIM`` in the short-range file and lower orders with longer ``S_MAXIM`` in the long-range file, then adjust using the guidelines above and validation on forces, energies, and structural observables.

Two-layer construction (short + long). The user specifies ChIMES hyperparameters independently for each layer: in particular, maximum bodiedness, maximum polynomial orders (:math:`\mathcal{O}_{n\mathrm{B}}`), and outer cutoffs :math:`r_{\mathrm{cut}}`. ChIMES LSQ then builds a design matrix per layer, :math:`\mathbf{M}_{\mathrm{short}}` and :math:`\mathbf{M}_{\mathrm{long}}`, with the same number of rows (one per constraint row derived from the reference data :math:`\mathbf{X}_{\mathrm{DFT}}`) but different numbers of columns when the two layers use different bodiedness / polynomial orders. The matrices are concatenated horizontally and the combined linear system is solved in weighted least-squares form:

.. math::
   :label: eq-turbo-design

   \mathbf{w}
   \left[\mathbf{M}_{\mathrm{short}}\ \mathbf{M}_{\mathrm{long}}\right]
   \begin{bmatrix}
   \mathbf{c}_{\mathrm{short}} \\ \mathbf{c}_{\mathrm{long}}
   \end{bmatrix}
   =
   \mathbf{w}\,\mathbf{X}_{\mathrm{DFT}}

Here :math:`\mathbf{w}` denotes the usual weighting of rows (forces, energies, stresses, etc., as in standard ChIMES LSQ). The coefficient blocks :math:`\mathbf{c}_{\mathrm{short}}` and :math:`\mathbf{c}_{\mathrm{long}}` are the separate parameters for the short- and long-ranged layers. At run time, LAMMPS evaluates the two ChIMES pair contributions and combines them through ``hybrid/overlay``, as produced by the ALD/ChIMES tool chain.

For multilayer fits, set ``REGRESS_ALG = "dlasso"`` in ``config.py``. In practice, ``dlasso`` has been more reliable than SVD-based solvers for these design matrices because strong covariance among columns can cause SVD to discard coefficients that remain important for accurate forces and energies.

-------

============================
Example fit: Propane system
============================

.. Note::

   Example files live under:

   ``<al_driver repository>/examples/simple_iter_single_statepoint-lmp-test-turbo-test``

The bundled propane example is an iterative united-atom fit for liquid propane. The force field defines two carbon types, ``C1`` (terminal methyl carbons) and ``C2`` (central methylene carbon), so bonded and non-bonded pair rows in ``fm_setup.in`` can be assigned separately. Layer ``0`` is tuned for short-ranged intramolecular structure (bonded geometry and stiff repulsion within each propane molecule); layer ``1`` carries longer-ranged intermolecular van der Waals interactions between molecules in the periodic box. Reference data for this bundled example use LAMMPS as the “QM” reference method (``BULK_QM_METHOD = "LMP"``), so you must supply both ChIMES MD templates and classical reference LAMMPS inputs (see below).

------------------------------------------
Directory layout (relative to the example)
------------------------------------------

Compared with :ref:`page-basic`, the ``ALC-0_BASEFILES`` area contains two numbered setup files instead of a single ``fm_setup.in``. The ``LMPMD_BASEFILES`` and ``LMP_BASEFILES`` directories supply ChIMES fitting MD and reference LAMMPS jobs, respectively.

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

Role of ``0.fm_setup.in`` vs. ``1.fm_setup.in``. Layer ``0`` encodes the short-ranged ChIMES model (typically smaller outer cutoffs and/or higher polynomial orders where the potential is stiff). Layer ``1`` encodes the long-ranged model (larger cutoffs and a sparser Chebyshev expansion where the potential is smoother). Hyperparameters in each file should follow the guidance provided in the ChIMES LSQ manual and be reasonable for describing your target physical system.

``N_HYPER_SETS`` in ``config.py`` must equal the number of numbered ``*.fm_setup.in`` files in ``ALC-0_BASEFILES`` (2 in this example). If ``N_HYPER_SETS`` is omitted, the driver defaults to ``1`` and runs the standard single-layer workflow. The driver copies each numbered ``fm_setup.in`` into its ChIMES LSQ build, assembles one design matrix per file from the same ``traj_list.dat`` / ``*.xyzf`` training data, and expects one reduced parameter file per layer after regression. The first layer is written to ``0params.txt.reduced``, the second to ``1params.txt.reduced``, and so on.

ChIMES LSQ options in each ``fm_setup.in``. The two layers share the same training trajectory (``# TRJFILE #``), frame count (``# NFRAMES #``), atom-type table (``# TYPEIDX #`` / ``# NATMTYP #``), and three-body exclusion list. They differ in the Chebyshev polynomial orders and cutoffs that define each layer’s resolution:

* ``# PAIRTYP #`` — ``CHEBYSHEV`` followed by maximum 2-, 3-, and 4-body polynomial orders (:math:`\mathcal{O}_{n\mathrm{B}}`; remember that orders :math:`\geq 3` are entered as :math:`n+1` in ``fm_setup.in``, as in :ref:`page-basic`).
* ``# PAIRIDX #`` — pair-specific inner and outer cutoffs (``S_MINIM``, ``S_MAXIM``), grid spacing (``S_DELTA``), and Morse parameters for each atom-type combination.
* ``# FCUTTYP #`` — inner cutoff functional form (Tersoff-style in the example).

In the bundled propane files, layer ``0`` uses ``CHEBYSHEV 6 6 0`` with ``S_MAXIM`` of 2.75 Å on all pair rows, while layer ``1`` uses ``CHEBYSHEV 4 0 0`` with ``S_MAXIM`` of 7.0 Å. The ``# NLAYERS #`` line in each file is the standard ChIMES LSQ control variable (here ``1`` per file) and is not the same as the driver’s multilayer count; do not confuse it with ``N_HYPER_SETS``.

For remaining ``fm_setup.in`` fields (weights, stress/energy fitting flags, Coulomb options, etc.), follow the `ChIMES LSQ manual <https://chimes-lsq.readthedocs.io/en/latest/index.html>`_ and keep both layer files consistent except where you intentionally change resolution or cutoffs.

-------

------------------------------------------
Input files and ``config.py``
------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Key ALD settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``N_HYPER_SETS`` to the number of independent ``*.fm_setup.in`` layers you intend to fit (2 for the standard short + long propane example). The ALD then builds, solves, and post-processes that many layer-specific design matrices and parameter files on every fitting cycle.

Multilayer TurboChIMES requires LAMMPS as the MD driver. Set ``MD_STYLE = "LMP"`` and point ``MD_MPI`` / ``CHIMES_MD_MPI`` at a LAMMPS executable built with the ChIMES calculator. The fitted potential is expressed as multiple ``chimesFF`` instances combined with ``hybrid/overlay``.

A minimal excerpt from the example ``config.py`` (compare full file in the example directory):

.. code-block:: python
   :linenos:
   :emphasize-lines: 26-27, 40-41

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

Replace all paths, queues, modules, and account settings with those appropriate for your HPC environment. The reference LAMMPS job in ``LMP_BASEFILES`` uses molecular atom styles and class2 angles in the bundled example; your ``LMP_EXE`` must be built with compatible packages (see :ref:`page-basic` LAMMPS discussion where applicable).

------------------------------------------
LAMMPS input for multilayer ChIMES
------------------------------------------

Each fitted layer maps to one ``chimesFF`` instance in LAMMPS. The number of ``chimesFF`` keywords in ``pair_style`` must match the number of ``*.fm_setup.in`` files used in the fit. The bundled propane MD template uses:

.. code-block:: text

   pair_style      hybrid/overlay chimesFF for_fitting chimesFF for_fitting
   pair_coeff      * * chimesFF  1  0params.txt.reduced
   pair_coeff      * * chimesFF  2  1params.txt.reduced

The integer after ``chimesFF`` on each ``pair_coeff`` line selects the overlay instance. The parameter file name must match the layer index produced by ChIMES LSQ post-processing.

-------

------------------------------------------
Running and inspecting output
------------------------------------------

The execution pattern matches :ref:`page-basic`:

1. Edit ``config.py`` and all paths under ``ALL_BASE_FILES/``.
2. From ``WORKING_DIR``, run the driver, e.g. ``python3 -u /path/to/al_driver/src/main.py 0 1 2 3`` (or your site’s wrapper), preferably inside ``screen`` or ``tmux``.

Inspect ChIMES LSQ logs, fitted parameter files for both layers, and subsequent LAMMPS trajectories as you would for a single-layer fit. If regression fails or weights are imbalanced, revisit ``WEIGHTS_*``, ``REGRESS_VAR``, ``REGRESS_ALG``, and the hyperparameters in ``0.fm_setup.in`` / ``1.fm_setup.in``.

-------

========================================================
Further reading and cross-links
========================================================

* ChIMES LSQ manual: `ChIMES LSQ documentation <https://chimes-lsq.readthedocs.io/en/latest/index.html>`_.
* LAMMPS ``hybrid/overlay`` with multiple ``chimesFF`` layers: see the ChIMES calculator documentation for your build.
* Single-layer baseline workflow: :ref:`page-basic`.
