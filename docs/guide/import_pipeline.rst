The Import Pipeline
================================================================================

The HATS import pipeline is a map-reduce style data processing pipeline designed
to convert large astronomical catalogs into the HATS format. It is optimized for
handling very large datasets that may not fit into memory, leveraging parallel
processing techniques to efficiently process and organize the data.

.. mermaid::
   :alt: Import Pipeline Flowchart

    graph TD
        run[run: Catalog creation pipeline]
        run --> setup[Setup: ResumePlan & pickle reader]
        
        setup --> check_mapping{should_run_mapping?}
        check_mapping -->|yes| mapping_submit[Submit mapping futures]
        check_mapping -->|no| binning_start
        
        mapping_submit -.parallel.-> map_tasks["mr.map_to_pixels (N input files)"]
        map_tasks --> map_wait[wait_for_mapping]
        map_wait --> binning_start
        
        binning_start[Binning Phase]
        binning_start --> read_hist[resume_plan.read_histogram]
        read_hist --> get_align[resume_plan.get_alignment_file]
        
        get_align --> check_align_exists{alignment file exists?}
        check_align_exists -->|yes| load_align[Load existing alignment]
        check_align_exists -->|no| check_constant{constant_healpix_order >= 0?}
        
        check_constant -->|yes| const_align[Create constant-order alignment]
        check_constant -->|no| gen_align[pixel_math.generate_alignment]
        
        const_align --> save_align[Save alignment to pickle]
        gen_align --> save_align
        load_align --> build_map[Build destination_pixel_map]
        save_align --> build_map
        
        build_map --> check_splitting
        
        check_splitting{should_run_splitting?}
        check_splitting -->|yes| split_submit[Submit splitting futures]
        check_splitting -->|no| check_reducing
        
        split_submit -.parallel.-> split_tasks["mr.split_pixels (N input files)"]
        split_tasks --> split_wait[wait_for_splitting]
        split_wait --> check_reducing
        
        check_reducing{should_run_reducing?}
        check_reducing -->|yes| reduce_submit[Submit reducing futures]
        check_reducing -->|no| check_finishing
        
        reduce_submit -.parallel.-> reduce_tasks["mr.reduce_pixel_shards (N destination pixels)"]
        reduce_tasks --> reduce_wait[wait_for_reducing]
        reduce_wait --> check_finishing
        
        check_finishing{should_run_finishing?}
        check_finishing -->|yes| finishing[Finishing Phase]
        check_finishing -->|no| done[Done]
        
        finishing --> done
        
        classDef mappingStyle fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
        classDef binningStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
        classDef splittingStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
        classDef reducingStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
        classDef finishingStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
        
        class mapping_submit,map_tasks,map_wait mappingStyle
        class binning_start,read_hist,get_align,check_align_exists,load_align,check_constant,const_align,gen_align,save_align,build_map binningStyle
        class split_submit,split_tasks,split_wait splittingStyle
        class reduce_submit,reduce_tasks,reduce_wait reducingStyle
        class finishing finishingStyle


Pipeline Stages
--------------------------------------------------------------------------------
The main script ``run_import.py`` executes a multi-part pipeline to create a
HATS catalog:

1. **Mapping Stage**:
    - Maps input files to Healpix pixels.
    - Uses the ``map_to_pixels`` function to process each input file and
      generate intermediate mapping results.
    - Writes partial results to the ``resume_path``.

2. **Binning Stage**:
    - Reads the raw histogram of mapped data using ``read_histogram``.
    - Validates the total number of rows against the expected count.
    - Generates an alignment file using ``get_alignment_file``, which maps
      high-order pixels to lower-order ones based on thresholds.

3. **Splitting Stage**:
    - Splits the mapped data into shards for further processing.
    - Uses the ``split_pixels`` function to process each input file and generate
      intermediate split results.
    - Writes results to the ``resume_path`` and ``cache_shard_path``.

4. **Reducing Stage**:
    - Reduces pixel shards into final catalog partitions.
    - Uses the ``reduce_pixel_shards`` function to aggregate data into
      destination pixels.
    - Writes the final output to the catalog path.

5. **Finishing Stage**:    
    - Writes metadata for the catalog, including partition information and
      column properties.
    - Optionally writes a skymap if specified.
    - Cleans up intermediate files and validates the catalog.
