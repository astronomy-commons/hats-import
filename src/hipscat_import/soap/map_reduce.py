"""Inner methods for SOAP"""

import os  # # TODO

import healpy as hp
import numpy as np
import pandas as pd
from hipscat.catalog import Catalog
from hipscat.io.paths import pixel_catalog_file
from hipscat.pixel_math.healpix_pixel import HealpixPixel
from hipscat.pixel_math.pixel_margins import get_margin
from hipscat.pixel_tree.pixel_alignment import PixelAlignment, align_trees


def source_to_object_map(args):
    """Build a map of (source order/pixel) to the (object order/pixel)
    that are aligned.
    """
    object_catalog = Catalog.read_from_hipscat(args.object_catalog_dir)
    source_catalog = Catalog.read_from_hipscat(args.source_catalog_dir)

    ## Direct aligment from source to object
    ###############################################
    alignment = align_trees(
        object_catalog.pixel_tree, source_catalog.pixel_tree, "outer"
    )
    som = alignment.pixel_mapping
    som = som.groupby(
        [PixelAlignment.JOIN_ORDER_COLUMN_NAME, PixelAlignment.JOIN_PIXEL_COLUMN_NAME],
        group_keys=False,
    )

    ## Lots of cute comprehension is happening here.
    ## create tuple of (source order/pixel) and [array of tuples of (object order/pixel)]
    source_to_object = [
        (
            HealpixPixel(source_name[0], source_name[1]),
            [
                HealpixPixel(object_elem[0], object_elem[1])
                for object_elem in object_group.dropna().to_numpy().T[:2].T
            ],
        )
        for source_name, object_group in som
    ]
    ## Treat the array of tuples as a dictionary.
    source_to_object = dict(source_to_object)

    ## Object neighbors for source
    ###############################################
    max_order = max(
        object_catalog.partition_info.get_highest_order(),
        source_catalog.partition_info.get_highest_order(),
    )

    object_order_map = np.full(hp.order2npix(max_order), -1)

    for pixel in object_catalog.partition_info.get_healpix_pixels():
        explosion_factor = 4 ** (max_order - pixel.order)
        exploded_pixels = [
            *range(
                pixel.pixel * explosion_factor,
                (pixel.pixel + 1) * explosion_factor,
            )
        ]
        object_order_map[exploded_pixels] = pixel.order

    source_to_neighbor_object = {}

    for source, objects in source_to_object.items():
        # get all neighboring pixels
        nside = hp.order2nside(source.order)
        neighbors = hp.get_all_neighbours(nside, source.pixel, nest=True)

        ## get rid of -1s and normalize to max order
        explosion_factor = 4 ** (max_order - source.order)
        neighbors = [
            neighbor * explosion_factor for neighbor in neighbors if neighbor != -1
        ]

        neighbors_orders = object_order_map[neighbors]
        desploded = [
            HealpixPixel(order, hoo_pixel >> 2 * (max_order - order))
            for order, hoo_pixel in list(zip(neighbors_orders, neighbors))
            if order != -1
        ]
        desploded = set(desploded) - set(objects)
        source_to_neighbor_object[source] = list(desploded)

    return source_to_object, source_to_neighbor_object


def _count_joins_for_object(
    source_data, object_catalog_dir, object_id_column, object_pixel
):
    object_path = pixel_catalog_file(
        catalog_base_dir=object_catalog_dir,
        pixel_order=object_pixel.order,
        pixel_number=object_pixel.pixel,
    )
    object_data = pd.read_parquet(
        path=object_path, columns=[object_id_column]
    ).set_index(object_id_column)

    joined_data = source_data.merge(
        object_data, how="inner", left_index=True, right_index=True
    )

    return len(joined_data)


def _write_count_results(cache_path, source_healpix, results):
    """Build a nice dataframe with pretty columns and rows"""
    num_results = len(results)
    dataframe = pd.DataFrame(results, columns=["Norder", "Npix", "num_rows"])

    dirs = [int(order / 10_000) * 10_000 if order >=0 else -1 for order, _, _ in results]

    dataframe["Dir"] = dirs
    dataframe["join_Norder"] = np.full(
        num_results, fill_value=source_healpix.order, dtype=np.int32
    )
    dataframe["join_Dir"] = int(dataframe["join_Norder"] / 10_000) * 10_000
    dataframe["join_Npix"] = np.full(
        num_results, fill_value=source_healpix.pixel, dtype=np.int32
    )
    dataframe = dataframe[
        ["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix", "num_rows"]
    ]
    dataframe.to_csv(
        os.path.join(cache_path, f"{source_healpix.order}_{source_healpix.pixel}.csv")
    )


def count_joins(
    args, source_healpix, object_order_pixels, object_neighbors, cache_path
):
    """Count the number of equijoined sources in the object pixels.

    If any un-joined source pixels remain, stretch out to neighboring object pixels.
    """
    source_path = pixel_catalog_file(
        catalog_base_dir=args.source_catalog_dir,
        pixel_order=source_healpix.order,
        pixel_number=source_healpix.pixel,
    )
    source_data = pd.read_parquet(
        path=source_path, columns=[args.source_object_id_column]
    ).set_index(args.source_object_id_column)

    remaining_sources = len(source_data)
    results = []
    objects_to_join = object_order_pixels + object_neighbors

    for object_pixel in objects_to_join:
        if remaining_sources < 1:
            break
        join_count = _count_joins_for_object(
            source_data,
            args.object_catalog_dir,
            args.object_id_column,
            object_pixel,
        )
        results.append([object_pixel.order, object_pixel.pixel, join_count])
        remaining_sources -= join_count

    ## mark that some sources were not joined
    if remaining_sources > 0:
        results.append([-1, -1, remaining_sources])

    _write_count_results(cache_path, source_healpix, results)
