Temporary files and disk usage
===============================================================================

This page aims to characterize intermediate files created by the hats-import 
catalog creation process. Most users are going to be ok with setting the ``tmp_dir``
and not thinking much more about it.

This page will mostly be useful for more contrained use cases, and when users have some
understanding when to use local vs remote object storage. This can also be a guide for
how to allocate resources for an import process, when the files will be large.

Overview
-------------------------------------------------------------------------------

In catalog creation, we currently have 3 different kinds of intermediate files:

- dask worker space (``dask_tmp``)

  - this is something dask needs. we don't really control this too much, nor should we.
  - when dask workers run out of memory, they spill their state to disk to be resumed by another worker.

- sharded parquet files (``tmp_dir``)

  - we split the input files by their destination pixel into smaller parquet files. 
    these small files are then concatenated into the final partitioned parquet files.

- intermediate resume files (``resume_tmp``)

  - log that some tasks are complete and their results. useful in the case of externally-killed jobs.

If you don't provide any of these arguments, we'll create an ``intermediate`` directory
under the ``output_path`` to shove temporary files. We also try to make reasonable
guesses if you only provide a subset of the arguments. If you want to control each
set of temporary files, then you should specify all three arguments.

There are also a few types of storage, and I'm going to give them arbitrary names:

- COLD

  - cloud object store

    - long-lived, potentially slow reads, potentially expensive reads and at-rest

  - big long term disk

    - something like a raid array. long-lived, but ideally faster and cheaper than object store.

- WARM

  - small regular local disk
  - something like a home directory, which might not have much room but is long-term

- HOT

  - short-term local disk

    - when running in cloud or HPC, this is the disk space that is associated with 
      the running instance. it generally has less space than your larger data allocations, 
      only lives for the duration of the job.
    - or when running on a regular linux system, the /tmp/ directory.

Thinking about the handful of environments we have run on, we have different needs for 
different kinds of intermediate files:

==================================  ============  =========== ==============
environment style                   ``dask_tmp``  ``tmp_dir`` ``resume_tmp``
==================================  ============  =========== ==============
vanilla linux / developer laptop    COLD          COLD        COLD
github actions / unit tests         HOT           HOT         HOT
HPC (e.g. PSC)                      HOT           COLD        WARM
Cloud (e.g. AWS)                    HOT           COLD        WARM
==================================  ============  =========== ==============

Some more explanation:

- AFAIK, dask worker data is never used between executions of the same job, 
  so the files need not be long-lived. they can always live in the HOT zone.
- the sharded parquet files have the potential to exceed the disk limits of either 
  HOT or WARM, and so should go where you have the most disk space available to you. 
  in addition, you want them to stick around between jobs if your job is killed, so they 
  shouldn't be on short-term local disk, even if you have 2T of it.
- resume files are going to be smaller than the sharded parquet files, and there are 
  going to be many times fewer of them. because they're small, you don't want to write 
  them to cloud object storage. however, you want them to stick around between jobs 
  (because that's the whole point of them).
- if you're ok not being able to resume your job, then you should totally have the 
  option to use HOT storage for sharded parquet and resume files, even in PSC/AWS scenarios. 
  you do you.

What's happening when
-------------------------------------------------------------------------------

The hats-import catalog creation process generates a lot of temporary files. Some find this 
surprising, so we try to provide a narrative of what's happening and why.

Planning stage
...............................................................................

At this stage, generally the only file that is written out is ``<resume_tmp>/input_paths.txt``
and contains the paths of all the input files. This is to make sure that resumed instances
of the job are using the same input files.

The final output directory has been created, but is empty until "Reducing".

Mapping stage
...............................................................................

In this stage, we're reading each input file and building a map of how many objects are in 
each high order pixel. For each input file, once finished, we will write a binary file with 
a sparse array representing the number of objects in each pixel. 

Binning stage
...............................................................................

In this stage, we're reading each of the numpy binary files from above, combining into a 
single histogram, and doing Some Math on it. 
We write out one new histogram file, and remove the intermediate histograms.

Splitting stage
...............................................................................

Here's where it starts to get serious.

In this stage, we re-read *all* the original input files. 
We have calculated our desired partitioning, and so we split each input file into shards 
based on their destination pixel. These are written out as parquet files. 

Say we have 1,000 input files and 4,000 destination pixels. We chunk the reads of input 
files to avoid out of memory issues, so say there are 1-30 chunks per file. We will 
have anywhere from 4,000 to 120,000,000 parquet shard files. 
There are levers to tune this, but since these are just intermediate files, 
it's likely not useful, unless you plan to run the import dozens of times over similar input data. 

If you're interested in tuning these levers, please reach out! We'd love to help! :doc:`/guide/contact`.

The total size on disk of the sharded parquet files is likely going to be within 
1-25% of the final catalog size on disk. You might expect it to be the same as the 
final catalog size, since it's the same data and it's all parquet. 
There are some subtleties that could cause a discrepancy:

- Additional file overhead of parquet metadata (more small files means more overhead)
- Different compression rates for small sets of points vs larger sets

Reducing stage
...............................................................................
In this stage, we're taking the sharded parquet files from the previous stage and combining 
them into a single parquet file per destination pixel.

For the example, we will have 4,000 tasks, each of which will concatenate the shard files 
for a single pixel into 4,000 final files. 
As the final files are written, the parquet shards are removed. 
This is when storage shifts from intermediate files to the real output files.

Finishing stage
...............................................................................

Here, we will write out a few additional final files (e.g. ``properties``, ``_metadata``).
Additionally, we will clean up any straggling intermediate resume files. 
This includes all text log files, and the summed histogram file. 
After this stage, we should have zero intermediate files.

Approximate file sizes
-------------------------------------------------------------------------------

We discuss the size of intermediate files for a sample hipcat-import pipeline, 
using the publicly available TIC catalog (:doc:`public/tic`).

We have 90 input files, all gzipped csvs. They range from 24M to 10G, and total 389G.

Total directory size (all intermediate files) and size in final output directory after each stage:

============= ====== ======================= =========== =========================
Stage         Intermediate                   Output
------------- ------------------------------ -------------------------------------
Stage         Size   Description             Size        Description
============= ====== ======================= =========== =========================
Planning      7.8 K  a single text log       0           it's created, but empty.
Mapping       8.5 G  all numpy histograms    0
Binning       97 M   one histogram, few logs 0
Splitting     535 G  246,227 files.
                     246,220 sharded parquet 0
Reducing      97 M   one histogram, few logs 439 G       3,768 parquet files
Finishing     0                              439 G       adds root-level metadata
============= ====== ======================= =========== =========================

Input vs output size
-------------------------------------------------------------------------------

As alluded to in the `Splitting` section above, the total on-disk size of the 
final catalog can be very different from the on-disk size of the input files.

In our internal testing, we converted a number of different kinds of catalogs, 
and share some of the results with you, to give some suggestion of the disk requirements
you may face when converting your own catalogs to hats format.

============= =============== =========== =============== =========================
Catalog	      Input size (-h) Input size  HATS size       Ratio
============= =============== =========== =============== =========================
allwise       1.2T             1196115700       310184460   0.26 (a lot smaller)
neowise	      3.9T             4177447284      4263269112   1.02 (about the same)
tic           389G              407367196       463070176   1.14 (a little bigger)
sdss (object) 425G              445204544       255775912   0.57 (a little smaller)
zubercal      8.4T             8991524224     11629945932   1.29 (a little bigger)
============= =============== =========== =============== =========================

Notes:

- allwise, neowise, and tic were all originally compressed CSV files.
- sdss was originally a series of fits files
- zubercal was originally 500k parquet files, and is reduced in the example to 
  around 70k hats parquet files.
